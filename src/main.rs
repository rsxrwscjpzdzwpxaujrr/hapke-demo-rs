use std::sync::RwLock;
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
use tiff::decoder::DecodingResult;
use wide::f32x8;
use crate::oren_nayar::{OrenNayar, OrenNayarParams};
use gtk::prelude::*;
use gtk::{gio, Application};
use gtk::glib::clone;
use crate::window::Window;

mod gl_area;
mod window;
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

const APP_ID: &str = "xyz.rsxrwscjpzdzwpxaujrr.hapke_demo_rs";
const THREAD_COUNT: usize = 2;
const MAP_WIDTH: usize = 360;
const MAP_HEIGHT: usize = 180;
const SIMD_SIZE: usize = 8;
const CHANNELS: usize = 3;
const BUFFER_SIZE: usize = (MAP_WIDTH / THREAD_COUNT) * MAP_HEIGHT * CHANNELS;

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

#[derive(PartialEq, Copy, Clone, Debug)]
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
    main_buffers: Vec<Arc<DoubleBuffer<Vec<u8>>>>,
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
            debug_str: Default::default(),
            normalized_albedo: RwLock::new(Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT)),
            on_params: RwLock::new(Vec::with_capacity(MAP_WIDTH * MAP_HEIGHT)),
            avg_time: Averager::new(Duration::from_secs_f32(0.5)),
            avg_calc_time: Averager::new(Duration::from_secs_f32(0.5)),
            main_buffers: Default::default(),
        };

        for _thread_id in 0..THREAD_COUNT {
            let mut buffer = Vec::with_capacity(BUFFER_SIZE);

            for _index in 0..BUFFER_SIZE {
                buffer.push(0);
            }

            data.main_buffers.push(Arc::new(DoubleBuffer::from(buffer)));
        }

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

fn gen_threads(data: Arc<Data>) -> Vec<(Arc<DoubleBuffer<Vec<u8>>>, JoinHandle<()>)> {
    const THREAD_WIDTH: usize = MAP_WIDTH / THREAD_COUNT;
    const ARR_SIZE: usize = ((MAP_HEIGHT * THREAD_WIDTH) - SIMD_SIZE) / SIMD_SIZE;

    (0..THREAD_COUNT).into_iter().map(|thread_id| {
        let data = data.clone();

        let mut buffer = Vec::with_capacity(BUFFER_SIZE);

        for _index in 0..BUFFER_SIZE {
            buffer.push(0);
        }

        let buffer_out = data.main_buffers[thread_id].clone();

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

                let albedo = &data.normalized_albedo.read().unwrap();

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

                let on_params = &data.on_params.read().unwrap();

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

                let light = data.light.read().unwrap().clone();
                let camera = data.camera.read().unwrap().clone();
                let mode = *(data.mode.read().unwrap());
                let exposure = data.exposure.read().unwrap().clone();
                let cursor = data.cursor.read().unwrap().clone();

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
                    *data.debug_str.write().unwrap() = debugger.get();
                }

                data.avg_time.add_measurement(start_time.elapsed().as_secs_f32());
                data.avg_calc_time.add_measurement(calc_time.as_secs_f32());
            }
        });

        (buffer_out, handle)
    }).collect()
}

fn polar_from_screen_coord(x: i32, y: i32) -> Option<(f32, f32)> {
    let width: i32 = 1080;//window.size().0 as i32;
    let height: i32 = 1080;//window.size().1 as i32;

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
    // Load GL pointers from epoxy (GL context management library used by GTK).
    {
        #[cfg(target_os = "macos")]
        let library = unsafe { libloading::os::unix::Library::new("libepoxy.0.dylib") }.unwrap();
        #[cfg(all(unix, not(target_os = "macos")))]
        let library = unsafe { libloading::os::unix::Library::new("libepoxy.so.0") }.unwrap();
        #[cfg(windows)]
        let library = libloading::os::windows::Library::open_already_loaded("libepoxy-0.dll")
            .or_else(|_| libloading::os::windows::Library::open_already_loaded("epoxy-0.dll"))
            .unwrap();

        epoxy::load_with(|name| {
            unsafe { library.get::<_>(name.as_bytes()) }
                .map(|symbol| *symbol)
                .unwrap_or(std::ptr::null())
        });
    }

    gio::resources_register_include!("resources.gresource").unwrap();

    let app = Application::builder().application_id(APP_ID).build();

    let data = Arc::new(Data::new());

    let i = FRAC_PI_6;
    let e = 0.001;
    let g = FRAC_PI_6;

    calculate_normalized_albedo_map(data.normalized_albedo.write().unwrap().deref_mut(), &data.params, i, e, g);
    calculate_onparam_map(data.on_params.write().unwrap().deref_mut(), &data.params, i, e, g);

    //let mut tex: [[f32; 180]; 360] = [[0.5; 180]; 360];

    //triangle.update_texture(&tex);

    let threads = gen_threads(data.clone());

    app.connect_activate(clone!{
        #[strong] data,
        move |app|
            Window::new(app, data.clone()).present()
        }
    );

    app.run();
}
