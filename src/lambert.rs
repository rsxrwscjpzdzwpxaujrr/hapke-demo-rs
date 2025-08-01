use wide::{f32x8, CmpLe};
use crate::shader::{Shader, ValueDebugger};
use crate::SIMD_SIZE;
use crate::vec3::Vec3;

pub(crate) struct Lambert<const CHANNELS: usize> {
    params: [f32x8; CHANNELS]
}

impl<const CHANNELS: usize> Shader<f32x8, CHANNELS> for Lambert<CHANNELS> {
    fn new(params: [f32x8; CHANNELS]) -> Self {
        Self {
            params,
        }
    }

    fn brdf(
        &self,
        light: &Vec3<f32x8>,
        normal: &Vec3<f32x8>,
        camera: &Vec3<f32x8>,
        debugger: [Option<&ValueDebugger>; SIMD_SIZE]
    ) -> [f32x8; CHANNELS] {
        //normal.dot(&light.clone()).clamp(0.0, 1.0) * albedo

        // let i = f32::acos(light.mul(-1.0).dot(&normal.mul(1.0)));
        // let e = f32::acos(camera.mul(-1.0).dot(&normal.mul(1.0)));
        // let g = f32::acos(camera.mul(-1.0).dot(&light.mul(-1.0)));
        //
        // let mu0 = f32::cos(i);
        // let mu = f32::cos(e);
        //
        // mu0 * albedo

        let mu = -camera.dot(normal);
        let mu0 = -light.dot(normal);

        let mu_le_zero = mu.cmp_le(f32x8::from(0.0));
        let mu0_le_zero = mu0.cmp_le(f32x8::from(0.0));

        if (mu_le_zero | mu0_le_zero).all() {
            return [f32x8::from(0.0); CHANNELS];
        }

        let result = self.params.map(|albedo| mu0 * albedo);

        for i in 0..SIMD_SIZE {
            if let Some(debugger) = debugger[i] {
                debugger.assign_str(
                    format!("μ: {:.5}\nμ₀: {:.5}\n\nValue: {}",
                            mu.as_array_ref()[i],
                            mu0.as_array_ref()[i],
                            result[0].as_array_ref()[i]
                    ));
            }
        }

        result
    }
}
