use crate::vec3::Vec3;

pub(crate) trait Shader<T> {
    fn brdf(&self, light: &Vec3<f32>, normal: &Vec3<f32>, camera: &Vec3<f32>, params: &T) -> f32;
}