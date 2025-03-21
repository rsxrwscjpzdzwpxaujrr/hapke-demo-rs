use std::array::from_fn;
use std::f32::consts::{FRAC_PI_4, FRAC_PI_2};
use crate::shader::Shader;
use std::f32::consts::PI;
use std::fs::File;
use std::ops::DerefMut;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::thread::{sleep, JoinHandle};
use std::time::{Duration, Instant};

use crate::double_buffer::DoubleBuffer;
use crate::hapke::{Hapke, HapkeParams};
use crate::lambert::Lambert;
use crate::utils::{to_cartesian};
use crate::vec3::Vec3;
use egui::mutex::RwLock;
use egui::ViewportId;
use egui_backend::egui::FullOutput;
use egui_backend::sdl2::video::GLProfile;
use egui_backend::{egui, gl, sdl2};
use egui_backend::{sdl2::event::Event, DpiScaling, ShaderVersion};
use egui_sdl2_gl as egui_backend;
use sdl2::mouse::MouseButton;
use sdl2::video::{SwapInterval, Window};
use sdl2::Sdl;
use tiff::decoder::DecodingResult;

mod graphics;
//mod old;
mod double_buffer;
mod shader;
mod lambert;
mod hapke;
mod utils;
mod vec3;

const THREAD_COUNT: usize = 2;
type Buffer = [u8; (360 / THREAD_COUNT) * 180 * 3];

fn load_hapke<P: AsRef<Path>>(path: P) -> Box<[[HapkeParams; 180]; 360]> {
    let f = File::open(path).unwrap();
    let mut decoder = tiff::decoder::Decoder::new(f).unwrap();
    let image = decoder.read_image().unwrap();

    let image_data = match image {
        DecodingResult::F32(data) => data,
        _ => panic!(),
    };

    let mut data: [[HapkeParams; 180]; 360] = [[HapkeParams::default(); 180]; 360];

    let mut pos = 0;

    for j in (20..160).rev() {
        for i in 0..data.len() {
            data[i][j].w = image_data[pos + 0];
            data[i][j].b = image_data[pos + 1];
            data[i][j].c = image_data[pos + 2];
            data[i][j].Bc0 = image_data[pos + 3];
            data[i][j].hc = image_data[pos + 4];
            data[i][j].Bs0 = image_data[pos + 5];
            data[i][j].hs = image_data[pos + 6];
            data[i][j].theta = image_data[pos + 7];
            data[i][j].phi = image_data[pos + 8];

            pos += 9;
        }
    }
    
    Box::new(data)
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
}

impl Mode {
    fn default() -> Mode {
        Mode::Hapke
    }
}

fn init_window(sdl_context: &Sdl) -> Window {
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

    window
}

struct Data {
    normals: [[Vec3<f32>; 180]; 360],
    params: OnceLock<[Box<[[HapkeParams; 180]; 360]>; 3]>,
    light: RwLock<Vec3<f32>>,
    camera: RwLock<Vec3<f32>>,
    mode: RwLock<Mode>,
    exposure: RwLock<f32>,
}

impl Data {
    fn new() -> Data {
        Data {
            params: OnceLock::from([ load_hapke("hapke_param_map_643nm.tif"),
                                     load_hapke("hapke_param_map_566nm.tif"),
                                     load_hapke("hapke_param_map_415nm.tif"), ]),
            light: RwLock::new([FRAC_PI_4.sin(), 0.0, FRAC_PI_4.sin()].into()),
            camera: RwLock::new([0.0, 0.0, 1.0].into()),
            mode: RwLock::new(Mode::default()),
            exposure: RwLock::new(1.0),
            normals: from_fn(|i| {
                from_fn(|j| {
                    let x = i;
                    let y = j;

                    let xf = x as f32;
                    let yf = y as f32 - 90.0;

                    let torad = PI / 180.0;
                    
                    let vector = to_cartesian(xf * torad, yf * torad);

                    vector
                })
            }),
        }
    }
}

fn gen_threads(data: Arc<Data>) -> Vec<(Arc<DoubleBuffer<Buffer>>, JoinHandle<()>, impl Fn())> {
    (0..THREAD_COUNT).into_iter().map(|thread_id| {
        let data = data.clone();

        let buffer_out = Arc::new(DoubleBuffer::from([0; (360 / THREAD_COUNT) * 180 * 3]));

        let buffer = buffer_out.clone();

        let render_out = Arc::new(AtomicBool::new(true));

        let render = render_out.clone();

        let handle = thread::spawn(move || {
            let offset = thread_id;

            loop {
                render.store(false, Ordering::Relaxed);

                let light = data.light.read().clone();
                let camera = data.camera.read().clone();
                let params = data.params.get().unwrap().clone();
                let mode = *(data.mode.read());
                let exposure = data.exposure.read().clone();

                //let mut tex: Box<[u8; 180 * 180 * 3]> = unsafe { Box::new(MaybeUninit::uninit().assume_init()) };

                for j in 0..180 {
                    for i in (offset..360).step_by(THREAD_COUNT) {
                        let x = i;
                        let y = j;

                        let normal = &data.normals[x][y];
                        //

                        let values = match mode {
                            Mode::Lambert => {
                                [
                                    Lambert{}.brdf(&light, &normal, &camera, &params[0][x][y].w),
                                    Lambert{}.brdf(&light, &normal, &camera, &params[1][x][y].w),
                                    Lambert{}.brdf(&light, &normal, &camera, &params[2][x][y].w),
                                ]
                            },
                            Mode::Hapke => {
                                [
                                    Hapke{}.brdf(&light, &normal, &camera, &params[0][x][y]),
                                    Hapke{}.brdf(&light, &normal, &camera, &params[1][x][y]),
                                    Hapke{}.brdf(&light, &normal, &camera, &params[2][x][y]),
                                ]
                            },
                        };

                        //let values = ((value * 2.0_f32.powf(exposure)) * 255.0) as u8;
                        
                        let values = values.map(|value| ((value * 2.0_f32.powf(exposure)).powf(1.0 / 2.2) * 255.0) as u8);

                        buffer.get_mut()[(j * 180 + (i / THREAD_COUNT)) * 3 + 0] = values[0];
                        buffer.get_mut()[(j * 180 + (i / THREAD_COUNT)) * 3 + 1] = values[1];
                        buffer.get_mut()[(j * 180 + (i / THREAD_COUNT)) * 3 + 2] = values[2];
                    }
                }

                buffer.flip();

                while !render.load(Ordering::Relaxed) {
                    sleep(Duration::from_millis(1));
                }
            }
        });

        (buffer_out, handle, move || { render_out.store(true, Ordering::Relaxed) })
    }).collect()
}

fn polar_from_screen_coord(x: i32, y: i32, window: &Window) -> Option<(f32, f32)> {
    let width: i32 = window.size().0 as i32;
    let height: i32 = window.size().1 as i32;
    
    if y < height / 2 {
        let xusize = x / (width  / 360);
        let yusize = y / (height / 360);
        
        if xusize < 0 || xusize > 360 {
            None
        } 
        else if yusize < 0 || yusize > 180 {
            None
        }
        else {
            Some(((xusize as f32).to_radians(), ((yusize - 90) as f32).to_radians()))
        }
    } else {
        // todo
        None
    }
}

fn id_from_polar(phi: f32, theta: f32) -> (usize, usize) {
    (
        (phi.to_degrees() as usize).clamp(0, 360 - 1),
        ((theta.to_degrees() + 90.0) as usize).clamp(0, 180 - 1),
    )
}

fn main() {
    let sdl_context = sdl2::init().unwrap();

    let window = init_window(&sdl_context);

    // Create a window context
    let _ctx = window.gl_create_context().unwrap();
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

    //let mut tex: [[f32; 180]; 360] = [[0.5; 180]; 360];
    
    //triangle.update_texture(&tex);
    
    let threads = gen_threads(data.clone());

    let mut triangle = graphics::Graphics::new();

    let start_time = Instant::now();
    
    let mut camera_polar = (-FRAC_PI_2, 0.0f32);

    while !quit {
        egui_state.input.time = Some(start_time.elapsed().as_secs_f64());
        egui_ctx.begin_frame(egui_state.input.take());

        // An example of how OpenGL can be used to draw custom stuff with egui
        // overlaying it:
        // First clear the background to something nice.
        unsafe {
            // Clear the screen to green
            gl::ClearColor(0.8, 0.8, 0.8, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        threads.iter().enumerate().for_each(|(i, (buffer, _, _))| {
            triangle.update_texture(***buffer, i);
        });

        // Then draw our triangle.
        triangle.draw(camera_polar.0, camera_polar.1);

        let e_window = {
            let light = &data.light.write();
            let camera = &data.camera.write();
            let mut exposure = data.exposure.write();
            let mut mode = data.mode.write();
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
                    let normal = data.normals[i][j].clone();
                    ui.label(format!("Normal: {}", normal));
                }

                ui.label(" ");

                ui.label("Select shader:");
                ui.selectable_value(mode.deref_mut(), Mode::Lambert, "Lambert");
                ui.selectable_value(mode.deref_mut(), Mode::Hapke, "Hapke");
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
        
        unsafe { gl::Disable(gl::DEPTH_TEST) };

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
                let vector = -to_cartesian(phi, theta);

                mouse.pressed_mouse_buttons().for_each(|button| {
                    let mut light = data.light.write();
                    let mut camera = data.camera.write();
                    
                    if button == MouseButton::Right {
                        camera_polar = (phi, theta);
                    }

                    *(match button {
                        MouseButton::Left => light,
                        MouseButton::Right => camera,
                        _ => unimplemented!(),
                    }) = vector.clone();
                });
            }
        }

        threads.iter().for_each(|(_, _, render)| {
            render()
        });
    }

    //thread.join();
}