use std::f32::consts::PI;
use wide::{f32x8, CmpLe};
use crate::shader::{Shader, ValueDebugger};
use crate::vec3::Vec3;

#[derive(Copy, Clone, Default)]
pub(crate) struct OrenNayarParams<T> {
    pub albedo: T,
    pub roughness: T,
}

impl From<[OrenNayarParams<f32>; 8]> for OrenNayarParams<f32x8> {
    fn from(value: [OrenNayarParams<f32>; 8]) -> Self {
        Self {
            albedo: f32x8::from(value.map(|value| value.albedo)),
            roughness: f32x8::from(value.map(|value| value.roughness)),
        }
    }
}

pub(crate) struct OrenNayar<const CHANNELS: usize> {
    params: [OrenNayarParams<f32x8>; CHANNELS],
}

impl<const CHANNELS: usize> Shader<OrenNayarParams<f32x8>, CHANNELS> for OrenNayar<CHANNELS> {
    fn new(params: [OrenNayarParams<f32x8>; CHANNELS]) -> Self {
        Self {
            params
        }
    }

    fn brdf(
        &self,
        light: &Vec3<f32x8>,
        normal: &Vec3<f32x8>,
        camera: &Vec3<f32x8>,
        debugger: [Option<&ValueDebugger>; 8]
    ) -> [f32x8; CHANNELS] {
        let mu = -camera.dot(normal);
        let mu0 = -light.dot(normal);

        let mu_le_zero = mu.cmp_le(f32x8::from(0.0));
        let mu0_le_zero = mu0.cmp_le(f32x8::from(0.0));

        if (mu_le_zero | mu0_le_zero).all() {
            return [f32x8::from(0.0); CHANNELS];
        }

        let normal: Vec3<f32x8> = [-normal.x, -normal.y, -normal.z].into();

        let s = light.dot(camera) - normal.dot(light) * normal.dot(camera);

        let t = s.cmp_le(f32x8::from(0.0)).blend(
            f32x8::from(1.0),
            normal.dot(light).max(normal.dot(camera)),
        );

        let result = self.params.map(|params| {
            let a =              1.0 / (PI + (PI / 2.0 - 2.0 / 3.0) * params.roughness);
            let b = params.roughness / (PI + (PI / 2.0 - 2.0 / 3.0) * params.roughness);

            normal.dot(light) * (a + (b * (s / t))) * params.albedo * PI
        });

        for i in 0..8 {
            if let Some(debugger) = debugger[i] {
                debugger.assign_str(
                    format!("μ: {:.5}\nμ₀: {:.5}\n\nRougness: {:.5}\ns: {:.5}\nt: {:.5}\n\nValue: {}",
                            mu.as_array_ref()[i],
                            mu0.as_array_ref()[i],
                            self.params[0].roughness.as_array_ref()[i],
                            s.as_array_ref()[i],
                            t.as_array_ref()[i],
                            result[0].as_array_ref()[i]
                    ));
            }
        }

        result
    }
}
