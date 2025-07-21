use crate::averager::Averager;
use crate::shader::{Shader, ValueDebugger};
use std::f32::consts::{FRAC_PI_3, FRAC_PI_6, PI};
use std::fs::File;
use std::ops::DerefMut;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::{array, thread};
use std::thread::{sleep, JoinHandle};
use std::time::{Duration, Instant};

use crate::double_buffer::DoubleBuffer;
use crate::hapke::{Hapke, HapkeParams};
use crate::lambert::Lambert;
use crate::utils::{to_cartesian, to_polar};
use crate::vec3::Vec3;
use egui::mutex::RwLock;
use egui::ViewportId;
use egui_backend::egui::FullOutput;
use egui_backend::sdl2::video::GLProfile;
use egui_backend::{egui, sdl2};
use egui_backend::{sdl2::event::Event, DpiScaling, ShaderVersion};
use egui_sdl2_gl as egui_backend;
use glow::HasContext;
use sdl2::mouse::MouseButton;
use sdl2::video::{GLContext, SwapInterval, Window};
use sdl2::Sdl;
use tiff::decoder::DecodingResult;
use wide::f32x8;
use crate::oren_nayar::{OrenNayar, OrenNayarParams};

mod graphics;
//mod old;
mod double_buffer;
mod shader;
mod lambert;
mod hapke;
mod utils;
mod vec3;
mod averager;
mod oren_nayar;

const THREAD_COUNT: usize = 2;
const MAP_WIDTH: usize = 360;
const MAP_HEIGHT: usize = 180;
const SIMD_SIZE: usize = 8;
const CHANNELS: usize = 3;

type Buffer = [u8; (MAP_WIDTH / THREAD_COUNT) * MAP_HEIGHT * CHANNELS];

fn load_hapke<P: AsRef<Path>>(path: P) -> Vec<HapkeParams<f32>> {
    const TEXTURE_FIRST_ROW: usize = 20;
    const TEXTURE_LAST_ROW: usize = 160;

    let f = File::open(path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(f).unwrap();
    let image = decoder.read_image().unwrap();

    let image_data = match image {
        DecodingResult::F32(data) => data,
        _ => panic!(),
    };

    let mut data: Vec<HapkeParams<f32>> = Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT);

    for row in 0..MAP_HEIGHT {
        if row >= TEXTURE_FIRST_ROW && row < TEXTURE_LAST_ROW {
            for column in 0..MAP_WIDTH {
                let mut param = HapkeParams::default();

                let pos = ((TEXTURE_LAST_ROW - row - 1) * MAP_WIDTH + column) * 9;

                param.w = image_data[pos + 0];
                param.b = image_data[pos + 1];
                param.c = image_data[pos + 2];
                param.Bc0 = image_data[pos + 3];
                param.hc = image_data[pos + 4];
                param.Bs0 = image_data[pos + 5];
                param.hs = image_data[pos + 6];
                param.theta = image_data[pos + 7];
                param.phi = image_data[pos + 8];

                data.push(param);
            }
        } else {
            for _column in 0..MAP_WIDTH {
                data.push(HapkeParams::default());
            }
        }
    }

    data
}

// fn calc_normals() -> [[Array1<f32>; 180]; 360] {
//     let mut data: ;
//
//     for j in 0..180 {
//         for i in 0..(data.len() as usize) {
//             to_cartesian(xf * PI * 2.0, (-yf + 0.5) * PI);
//         }
//     }
//
//     data
// }

#[derive(PartialEq, Copy, Clone)]
enum Mode {
    Lambert,
    Hapke,
    OrenNayar,
}

impl Mode {
    fn default() -> Mode {
        Mode::Hapke
    }
}

fn init_window(sdl_context: &Sdl) -> (Window, glow::Context, GLContext) {
    let video_subsystem = sdl_context.video().unwrap();
    let gl_attr = video_subsystem.gl_attr();
    gl_attr.set_context_profile(GLProfile::Core);

    // Let OpenGL know we are dealing with SRGB colors so that it
    // can do the blending correctly. Not setting the framebuffer
    // leads to darkened, oversaturated colors.
    gl_attr.set_double_buffer(true);
    // gl_attr.set_multisample_samples(1);
    gl_attr.set_framebuffer_srgb_compatible(false);

    // OpenGL 3.2 is the minimum that we will support.
    gl_attr.set_context_version(3, 2);

    let window = video_subsystem
        .window(
            "Hapke demo",
            1080,
            1080,
        )
        .opengl()
        .build()
        .unwrap();

    let _ctx = window.gl_create_context().unwrap();

    let gl = unsafe { glow::Context::from_loader_function(|s| video_subsystem.gl_get_proc_address(s) as *const _) };

    (window, gl, _ctx)
}

struct Data {
    normals: Vec<Vec3<f32>>,
    params: [Vec<HapkeParams<f32>>; CHANNELS],
    light: RwLock<Vec3<f32>>,
    camera: RwLock<Vec3<f32>>,
    mode: RwLock<Mode>,
    exposure: RwLock<f32>,
    cursor: RwLock<(f32, f32)>,
    debug_str: RwLock<String>,
    normalized_albedo: RwLock<Vec<[f32; CHANNELS]>>,
    on_params: RwLock<Vec<[OrenNayarParams<f32>; CHANNELS]>>,
    avg_time: Averager,
    avg_calc_time: Averager,
}

impl Data {
    fn new() -> Data {
        let mut data = Data {
            params: [
                load_hapke("hapke_param_map_643nm.tif"),
                load_hapke("hapke_param_map_566nm.tif"),
                load_hapke("hapke_param_map_415nm.tif"),
            ],
            light: RwLock::new([-0.93847078, -0.32556817, 0.11522973].into()),
            camera: RwLock::new([-0.8171755, 0.22495106, -0.53068].into()),
            mode: RwLock::new(Mode::default()),
            exposure: RwLock::new(2.0),
            normals: Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT),
            cursor: RwLock::new((0.0f32, 0.0f32)),
            debug_str: RwLock::new(String::default()),
            normalized_albedo: RwLock::new(Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT)),
            on_params: RwLock::new(Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT)),
            avg_time: Averager::new(Duration::from_secs_f32(0.5)),
            avg_calc_time: Averager::new(Duration::from_secs_f32(0.5)),
        };

        for row in 0..MAP_HEIGHT {
            for column in 0..MAP_WIDTH {
                let x = column;
                let y = row;

                let xf = x as f32;
                let yf = y as f32 - 90.0;

                let torad = PI / 180.0;

                let vector = to_cartesian(xf * torad, yf * torad);

                data.normals.push(vector)
            }
        }

        data
    }
}

fn gen_threads(data: Arc<Data>) -> Vec<(Arc<DoubleBuffer<Buffer>>, JoinHandle<()>)> {
    const THREAD_WIDTH: usize = MAP_WIDTH / THREAD_COUNT;
    const ARR_SIZE: usize = ((MAP_HEIGHT * THREAD_WIDTH) - SIMD_SIZE) / SIMD_SIZE;

    (0..THREAD_COUNT).into_iter().map(|thread_id| {
        let data = data.clone();

        let buffer_out = Arc::new(
            DoubleBuffer::from([0; (MAP_WIDTH / THREAD_COUNT) * MAP_HEIGHT * CHANNELS])
        );

        let buffer = buffer_out.clone();

        let handle = thread::spawn(move || {
            let offset = thread_id;

            let get_coords = |id: usize| -> (usize, usize) {
                let x = (id % THREAD_WIDTH) * THREAD_COUNT + offset;
                let y = id / THREAD_WIDTH;

                (x, y)
            };

            let hapke_paramsx8 = {
                let mut paramsx8: [Vec<HapkeParams<f32x8>>; CHANNELS]
                    = array::from_fn(|_| Vec::with_capacity(ARR_SIZE));

                let params = &data.params;

                for channel in 0..CHANNELS {
                    for k in 0..ARR_SIZE {
                        paramsx8[channel].push(array::from_fn(|l| {
                            let (x, y) = get_coords(k * SIMD_SIZE + l);

                            params[channel][y * MAP_WIDTH + x]
                        }).into())
                    }
                }

                paramsx8
            };

            let paramsx8 = {
                let mut paramsx8: [Vec<f32x8>; CHANNELS]
                    = array::from_fn(|_| Vec::with_capacity(ARR_SIZE));

                let albedo = &data.normalized_albedo.read();

                for channel in 0..CHANNELS {
                    for k in 0..ARR_SIZE {
                        paramsx8[channel].push(array::from_fn(|l| {
                            let (x, y) = get_coords(k * SIMD_SIZE + l);

                            albedo[y * MAP_WIDTH + x][channel]
                        }).into())
                    }
                }

                paramsx8
            };

            let onparamsx8 = {
                let mut paramsx8: [Vec<OrenNayarParams<f32x8>>; CHANNELS]
                    = array::from_fn(|_| Vec::with_capacity(ARR_SIZE));

                let on_params = &data.on_params.read();

                for channel in 0..CHANNELS {
                    for k in 0..ARR_SIZE {
                        paramsx8[channel].push(array::from_fn(|l| {
                            let (x, y) = get_coords(k * SIMD_SIZE + l);

                            on_params[y * MAP_WIDTH + x][channel]
                        }).into())
                    }
                }

                paramsx8
            };

            let normalsx8 = {
                let mut paramsx8: Vec<Vec3<f32x8>> = Vec::with_capacity(ARR_SIZE);

                for k in 0..ARR_SIZE {
                    let vecs: [Vec3<f32>; SIMD_SIZE] = array::from_fn(|l| {
                        let (x, y) = get_coords(k * SIMD_SIZE + l);

                        data.normals[y * MAP_WIDTH + x]
                    });

                    paramsx8.push(Vec3::<f32x8>::from([
                        array::from_fn(|i| vecs[i].x).into(),
                        array::from_fn(|i| vecs[i].y).into(),
                        array::from_fn(|i| vecs[i].z).into()]
                    ));
                }

                paramsx8
            };

            loop {
                let start_time = Instant::now();
                let mut calc_time = Duration::default();

                let light = data.light.read().clone();
                let camera = data.camera.read().clone();
                let mode = *(data.mode.read());
                let exposure = data.exposure.read().clone();
                let cursor = data.cursor.read().clone();

                let mut buffer_w = buffer.get_mut();

                let mut our_debuggerx8: [Option<&ValueDebugger>; SIMD_SIZE] = Default::default();

                let debugger = ValueDebugger::default();

                let cursor_id = id_from_polar(cursor.0, cursor.1);
                let cursor_id = ((cursor_id.0 + MAP_HEIGHT) % MAP_WIDTH, cursor_id.1);

                let exposure = 2.0_f32.powf(exposure);

                for k in (0..((MAP_HEIGHT * THREAD_WIDTH) - SIMD_SIZE)).step_by(SIMD_SIZE) {
                    for l in 0..SIMD_SIZE {
                        let (x, y) = get_coords(k + l);

                        our_debuggerx8[l] = if x == cursor_id.0 && y == cursor_id.1 {
                            Some(&debugger)
                        } else {
                            None
                        };
                    }

                    let start_time = Instant::now();

                    let values: [f32x8; CHANNELS] = match mode {
                        Mode::Lambert => {
                            let shader = Lambert::new(array::from_fn(|channel|
                                paramsx8[channel][k / SIMD_SIZE])
                            );

                            shader.brdf(
                                &light.into(),
                                &normalsx8[k / SIMD_SIZE],
                                &camera.into(),
                                our_debuggerx8)
                        },
                        Mode::Hapke => {
                            let shader = Hapke::new(array::from_fn(|channel|
                                hapke_paramsx8[channel][k / SIMD_SIZE])
                            );

                            shader.brdf(
                                &light.into(),
                                &normalsx8[k / SIMD_SIZE],
                                &camera.into(),
                                our_debuggerx8)
                        },
                        Mode::OrenNayar => {
                            let shader = OrenNayar::new(array::from_fn(|channel|
                                onparamsx8[channel][k / SIMD_SIZE])
                            );

                            shader.brdf(
                                &light.into(),
                                &normalsx8[k / SIMD_SIZE],
                                &camera.into(),
                                our_debuggerx8)
                        },
                    };

                    calc_time += start_time.elapsed();

                    let values = values.map(|value|
                        value.to_array().map(|value| ((value * exposure).powf(1.0 / 2.2) * 255.0))
                    );

                    for l in 0..SIMD_SIZE {
                        for channel in 0..CHANNELS {
                            buffer_w[((k + l) * CHANNELS) + channel] = values[channel][l] as u8;
                        }
                    }
                }

                buffer.flip();

                if !debugger.empty() {
                    *data.debug_str.write() = debugger.get();
                }

                data.avg_time.add_measurement(start_time.elapsed().as_secs_f32());
                data.avg_calc_time.add_measurement(calc_time.as_secs_f32());
            }
        });

        (buffer_out, handle)
    }).collect()
}

fn polar_from_screen_coord(x: i32, y: i32, window: &Window) -> Option<(f32, f32)> {
    let width: i32 = window.size().0 as i32;
    let height: i32 = window.size().1 as i32;

    if y < height / 2 {
        let xusize = x / (width  / MAP_WIDTH as i32);
        let yusize = y / (height / MAP_WIDTH as i32);

        if xusize < 0 || xusize > MAP_WIDTH as i32 {
            None
        }
        else if yusize < 0 || yusize > MAP_HEIGHT as i32 {
            None
        }
        else {
            Some((
                ((xusize - (MAP_WIDTH as i32 / 2)) as f32).to_radians(),
                ((yusize - (MAP_HEIGHT as i32 / 2)) as f32).to_radians(),
            ))
        }
    } else {
        // todo
        None
    }
}

fn id_from_polar(phi: f32, theta: f32) -> (usize, usize) {
    (
        ((phi.to_degrees() + 180.0) as usize).clamp(0, MAP_WIDTH - 1),
        ((theta.to_degrees() + 90.0) as usize).clamp(0, MAP_HEIGHT - 1),
    )
}

fn from_spherical(i: f32, e: f32, g: f32) -> (Vec3<f32>, Vec3<f32>, Vec3<f32>) {
    let x = (g.cos() - e.cos() * i.cos()) / i.sin();
    let y = (e.sin() * e.sin() - x * x).sqrt();

    let light: Vec3<f32> = [i.sin(), 0.0, i.cos()].into();
    let normal: Vec3<f32> = [0.0, 0.0, -1.0].into();
    let camera: Vec3<f32> = [x, y, e.cos()].into();

    (light, normal, camera)
}

fn calculate_normalized_albedo_map(
    buffer: &mut Vec<[f32; CHANNELS]>,
    params: &[Vec<HapkeParams<f32>>; CHANNELS],
    i: f32, e: f32, g: f32
) {
    for row in 0..MAP_HEIGHT {
        for column in 0..MAP_WIDTH {
            let shader = Hapke::new(array::from_fn(|i|
                HapkeParams::<f32x8>::from([params[i][row * MAP_WIDTH + column]; SIMD_SIZE])
            ));

            let (light, normal, camera) = from_spherical(i, e, g);

            let value = shader.brdf_non_simd(
                &light,
                &normal,
                &camera,
                None
            );

            buffer.push(value.map(|value| value / i.cos()));
        }
    }
}

fn calculate_onparam_map(
    buffer: &mut Vec<[OrenNayarParams<f32>; CHANNELS]>,
    params: &[Vec<HapkeParams<f32>>; CHANNELS],
    i: f32, e: f32, g: f32
) {
    for row in 0..MAP_HEIGHT {
        for column in 0..MAP_WIDTH {
            let shader = Hapke::new(array::from_fn(|i|
                HapkeParams::<f32x8>::from([params[i][row * MAP_WIDTH + column]; SIMD_SIZE])
            ));

            let (light, normal, camera) = from_spherical(i, e, g);

            buffer.push(shader.brdf_non_simd(
                &light,
                &normal,
                &camera,
                None
            ).map(|value| OrenNayarParams {
                albedo: value * 1.3,
                roughness: 0.55,
            }));
        }
    }
}

fn main() {
    let sdl_context = sdl2::init().unwrap();

    let (window, gl, _ctx) = init_window(&sdl_context);

    // debug_assert_eq!(gl_attr.context_profile(), GLProfile::Core);
    // debug_assert_eq!(gl_attr.context_version(), (3, 2));

    // Enable vsync
    if let Err(error) = window.subsystem().gl_set_swap_interval(SwapInterval::VSync) {
        println!(
            "Failed to gl_set_swap_interval(SwapInterval::VSync): {}",
            error
        );
    };

    // Init egui stuff
    let (mut painter, mut egui_state) =
        egui_backend::with_sdl2(&window, ShaderVersion::Default, DpiScaling::Default);
    let egui_ctx = egui::Context::default();
    let mut event_pump: sdl2::EventPump = sdl_context.event_pump().unwrap();

    let mut quit = false;

    let data = Arc::new(Data::new());

    let i = FRAC_PI_6;
    let e = 0.001;
    let g = FRAC_PI_6;

    calculate_normalized_albedo_map(data.normalized_albedo.write().deref_mut(), &data.params, i, e, g);
    calculate_onparam_map(data.on_params.write().deref_mut(), &data.params, i, e, g);

    //let mut tex: [[f32; 180]; 360] = [[0.5; 180]; 360];

    //triangle.update_texture(&tex);

    let threads = gen_threads(data.clone());

    let mut triangle = graphics::Graphics::new(&gl);

    let start_time = Instant::now();

    while !quit {
        egui_state.input.time = Some(start_time.elapsed().as_secs_f64());
        egui_ctx.begin_frame(egui_state.input.take());

        // An example of how OpenGL can be used to draw custom stuff with egui
        // overlaying it:
        // First clear the background to something nice.
        unsafe {
            // Clear the screen to green
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        }

        threads.iter().enumerate().for_each(|(i, (buffer, _))| {
            if let Some(buffer) = buffer.read() {
                triangle.update_texture(&gl, buffer.as_ref(), i);
            }
        });

        let camera_polar = to_polar(*data.camera.read());

        // Then draw our triangle.
        triangle.draw(&gl, camera_polar.0, camera_polar.1);

        let e_window = {
            let light = &data.light.write();
            let camera = &data.camera.write();
            let mut exposure = data.exposure.write();
            let mut mode = data.mode.write();
            let debug_str = data.debug_str.read().clone();
            let mouse = event_pump.mouse_state();

            let (x, y) = (mouse.x(), window.size().1 as i32 - mouse.y());

            let polar = polar_from_screen_coord(x, y, &window);

            egui::Window::new("Parameters").show(&egui_ctx, |ui| {
                ui.set_min_width(350.0);

                ui.add(egui::Slider::new(exposure.deref_mut(), -8.0..=8.0).text("exposure"));

                ui.label(" ");

                ui.label(format!("Light vector: {}", light));
                ui.label(format!("Camera vector: {}", camera));

                if let Some((phi, theta)) = polar {
                    let (i, j) = id_from_polar(phi, theta);
                    let normal = data.normals[j * MAP_HEIGHT + i].clone();
                    ui.label(format!("Normal: {}", normal));
                }

                ui.label(" ");

                debug_str.lines().for_each(|line| { ui.label(line); });

                ui.label(" ");

                ui.label(format!("Time: {:.4} sec", data.avg_time.average()));
                ui.label(format!("Calc time: {:.4} sec", data.avg_calc_time.average()));

                ui.label(" ");

                ui.label("Select shader:");
                ui.selectable_value(mode.deref_mut(), Mode::Lambert, "Lambert");
                ui.selectable_value(mode.deref_mut(), Mode::Hapke, "Hapke");
                ui.selectable_value(mode.deref_mut(), Mode::OrenNayar, "Oren-Nayar");
                ui.label(" ");
                if ui.button("Quit").clicked() {
                    quit = true;
                }
            })
        };

        let gui_rect = e_window.unwrap().response.rect;

        let FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = egui_ctx.end_frame();
        // Process output
        egui_state.process_output(&window, &platform_output);

        unsafe { gl.disable(glow::DEPTH_TEST) };

        let paint_jobs = egui_ctx.tessellate(shapes, pixels_per_point);

        // Note: passing a bg_color to paint_jobs will clear any previously drawn stuff.
        // Use this only if egui is being used for all drawing and you aren't mixing your own Open GL
        // drawing calls with it.
        // Since we are custom drawing an OpenGL Triangle we don't need egui to clear the background.
        painter.paint_jobs(None, textures_delta, paint_jobs);

        window.gl_swap_window();

        let repaint_after = viewport_output
            .get(&ViewportId::ROOT)
            .expect("Missing ViewportId::ROOT")
            .repaint_delay;

        if !repaint_after.is_zero() {
            if let Some(event) = event_pump.wait_event_timeout(4) {
                match event {
                    Event::Quit { .. } => quit = true,
                    _ => {
                        // Process input event
                        egui_state.process_input(&window, event, &mut painter);
                    }
                }
            }
        } else {
            for event in event_pump.poll_iter() {
                match event {
                    _ => {
                        // Process input event
                        egui_state.process_input(&window, event, &mut painter);
                    }
                }
            }
        }

        let mouse = event_pump.mouse_state();

        if !gui_rect.contains((mouse.x() as f32, mouse.y() as f32).into()) {
            let polar = polar_from_screen_coord(mouse.x(), window.size().1 as i32 - mouse.y(), &window);

            if let Some((phi, theta)) = polar {
                let mut cursor = data.cursor.write();
                *cursor.deref_mut() = (phi, theta);

                let vector = -to_cartesian(phi, theta);

                mouse.pressed_mouse_buttons().for_each(|button| {
                    let mut light = data.light.write();
                    let mut camera = data.camera.write();

                    *(match button {
                        MouseButton::Left => light,
                        MouseButton::Right => camera,
                        _ => unimplemented!(),
                    }) = vector.clone();
                });
            }
        }
    }

    triangle.deinit(&gl);

    //thread.join();
}
