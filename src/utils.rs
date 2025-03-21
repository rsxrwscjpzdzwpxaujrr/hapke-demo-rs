use crate::vec3::Vec3;
pub(crate) fn to_cartesian(phi: f32, theta: f32) -> Vec3<f32> {
    [
        f32::cos(phi) * f32::cos(theta),
        f32::sin(theta),
        f32::sin(phi) * f32::cos(theta),
    ].into()
}
