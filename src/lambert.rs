use crate::shader::Shader;
use crate::vec3::Vec3;

pub(crate) struct Lambert {}

impl Shader<f32> for Lambert {
    fn brdf(&self, light: &Vec3<f32>, normal: &Vec3<f32>, camera: &Vec3<f32>, albedo: &f32) -> f32 {
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

        if mu <= 0.0 || mu0 <= 0.0 {
            return 0.0;
        }

        mu0 * albedo * 0.5
    }
}