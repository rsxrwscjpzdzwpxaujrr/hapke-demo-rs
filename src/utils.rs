use std::f32::consts::{FRAC_PI_2, PI};
use crate::vec3::Vec3;

pub(crate) fn to_cartesian(phi: f32, theta: f32) -> Vec3<f32> {
    [
        f32::cos(phi) * f32::cos(theta),
        f32::sin(theta),
        f32::sin(phi) * f32::cos(theta),
    ].into()
}

pub(crate) fn to_polar(cartesian: Vec3<f32>) -> (f32, f32) {
    let phi = f32::atan2(cartesian.z, cartesian.x) + PI;
    let theta = f32::acos(cartesian.y) - FRAC_PI_2;

    (phi, theta)
}
