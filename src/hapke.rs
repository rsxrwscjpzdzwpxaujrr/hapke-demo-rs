use std::f32::consts::PI;
use std::ops::Mul;
use wide::{f32x8, CmpEq, CmpLe};
use crate::shader::{Shader, ValueDebugger};
use crate::vec3::Vec3;

#[derive(Copy, Clone, Default)]
pub(crate) struct HapkeParams<T> {
    pub w    : T, // Single scattering albedo
    pub b    : T, // Henyey-Greenstein double-lobed single particle phase function parameter
    pub c    : T, // Henyey-Greenstein double-lobed single particle phase function parameter
    pub Bc0  : T, // Amplitude of Coherent Backscatter Opposition Effect (CBOE) - fixed at 0.0
    pub hc   : T, // Angular width of CBOE - fixed at 1.0
    pub Bs0  : T, // Amplitude of Shadow Hiding Opposition Effect (SHOE)
    pub hs   : T, // Angular width of SHOE
    pub theta: T, // Effective value of the photometric roughness - fixed at 23.657
    pub phi  : T, // Filling factor - fixed at 1.0
}

impl From<[HapkeParams<f32>; 8]> for HapkeParams<f32x8> {
    fn from(value: [HapkeParams<f32>; 8]) -> Self {
        Self {
            w: f32x8::from(value.map(|value| value.w)),
            b: f32x8::from(value.map(|value| value.b)),
            c: f32x8::from(value.map(|value| value.c)),
            Bc0: f32x8::from(value.map(|value| value.Bc0)),
            hc: f32x8::from(value.map(|value| value.hc)),
            Bs0: f32x8::from(value.map(|value| value.Bs0)),
            hs: f32x8::from(value.map(|value| value.hs)),
            theta: f32x8::from(value.map(|value| value.theta)),
            phi: f32x8::from(value.map(|value| value.phi)),
        }
    }
}

pub(crate) struct Hapke {}

fn acos_clamped(x: f32x8) -> f32x8 {
    f32x8::from(x.to_array().map(|x| x.clamp(-1.0, 1.0).acos()))
}

impl Shader<HapkeParams<f32x8>> for Hapke {
    fn brdf(&self, light: &Vec3<f32x8>, normal: &Vec3<f32x8>, camera: &Vec3<f32x8>, data: &HapkeParams<f32x8>, debugger: [Option<&ValueDebugger>; 8]) -> f32x8 {
        //let K = -f32::ln(1.0 - (1.209 * data.phi.powf(2.0 / 3.0))) / data.phi.powf(2.0 / 3.0);
        
        if data.w.cmp_eq(0.0).all() {
            return f32x8::from(0.0);
        }
        
        let mu = -camera.dot(normal);
        let mu0 = -light.dot(normal);

        let mu_le_zero = mu.cmp_le(f32x8::from(0.0));
        let mu0_le_zero = mu0.cmp_le(f32x8::from(0.0));

        if (mu_le_zero | mu0_le_zero).all() {
            return f32x8::from(0.0);
        }
        
        // if mu <= 0.0 || mu0 <= 0.0 {
        //     return 0.0;
        // }
        
        let tan_theta = f32x8::from(data.theta.to_radians().tan());
        let K = 1.0 - data.phi;
        
        let i = PI - acos_clamped(light.dot(normal));
        let e = PI - acos_clamped(camera.dot(normal));
        
        let g = acos_clamped(camera.dot(light));
        
        let Ei = e12(tan_theta, i);
        let Ee = e12(tan_theta, e);
        
        let phi = acos_clamped((camera.dot(light) - (mu * mu0)) / e.sin() * i.sin());
        //let phi = phi.clamp(0.0, FRAC_PI_2);
        
        //let phi = if phi.is_nan() { PI } else { phi };
        
        let temp_mask = i.cmp_le(e);

        let temp = {(
            temp_mask.blend(tmpfunc(phi.cos() ,  1.0, phi, Ei, Ee),
                            tmpfunc(1.0.into(), -1.0, phi, Ee, Ei)),
            temp_mask.blend(tmpfunc(1.0.into(), -1.0, phi, Ei, Ee),
                            tmpfunc(phi.cos() ,  1.0, phi, Ee, Ei))
        )};

        let x = 1.0 / (1.0 + (PI * pow2(tan_theta))).sqrt();
        
        let mu0_e = x * (mu0 + (i.sin() * tan_theta * temp.0));
        let mu_e  = x * (mu  + (e.sin() * tan_theta * temp.1));
        
        // let mu0_e = mu0;
        // let mu_e = mu;
        
        let ls = mu0_e / (mu_e + mu0_e);
        
        let p =
            ((1.0 + data.c) / 2.0) *
                (1.0 - (data.b * data.b)) / (1.0 - (2.0 * data.b * g.cos()) + pow2(data.b)).powf(3.0 / 2.0) +
                ((1.0 - data.c) / 2.0) *
                    (1.0 - (data.b * data.b)) / (1.0 + (2.0 * data.b * g.cos()) + pow2(data.b)).powf(3.0 / 2.0);
        
        let bs = 1.0 / (1.0 + ((g / 2.0).tan() / data.hs));
        
        let M = (compute_H(mu0_e / K, f32x8::from(data.w)) * compute_H(mu_e / K, f32x8::from(data.w))) - 1.0;
        
        let f = (-2.0 * (phi / 2.0).tan()).exp();
        
        //let f = if f == f32::INFINITY { 1.0 } else if f == f32::NEG_INFINITY { 0.0 } else {f};
        
        let ghi = gh(x, tan_theta, i);
        let ghe = gh(x, tan_theta, e);

        let temp = temp_mask.blend(mu0 / ghi, mu / ghe);
        // let temp = if i <= e {
        //     mu0 / ghi
        // }
        // else {
        //     mu / ghe
        // };
        
        let shadowing = (mu_e / ghe) * (mu0 / ghi) * (x / (1.0 - f + (f * x * temp)));
        
        // let shadowing = if shadowing.is_normal() { shadowing } else { 1.0 };
        
        let result = ls * K * (data.w / 4.0) * (p * (1.0 + data.Bs0 * bs) + M) * (1.0 + data.Bc0 * compute_Bc(g, f32x8::from(data.hc))) * shadowing;

        for j in 0..8 {
            if let Some(debugger) = debugger[j] {
                debugger.assign_str(
                    format!("i: {:.3}°\ne: {:.3}°\ng: {:.3}°\nψ: {:.3}°\n\nμ: {:.5}\nμ₀: {:.5}\n\nμe: {:.5}\nμ₀e: {:.5}\n\nShadowing: {:.5}\n\nValue: {}",
                            i.as_array_ref()[j].to_degrees(),
                            e.as_array_ref()[j].to_degrees(),
                            g.as_array_ref()[j].to_degrees(),
                            phi.as_array_ref()[j].to_degrees(),
                            mu.as_array_ref()[j],
                            mu0.as_array_ref()[j],
                            mu_e.as_array_ref()[j],
                            mu0_e.as_array_ref()[j],
                            shadowing.as_array_ref()[j],
                            result.as_array_ref()[j]
                    ));
            }
        }
        
        result
    }
}

fn tmpfunc(x1: f32x8, x2: f32, phi: f32x8, Ea: (f32x8, f32x8), Eb: (f32x8, f32x8)) -> f32x8 {
    ((x1 * Eb.1) + (pow2((phi / 2.0).sin()) * Ea.1) * x2) / (2.0 - Eb.0 - ((phi / PI) * Ea.0))
}

fn pow2(value: f32x8) -> f32x8 {
    value.mul(value)
}

fn e12(tan_theta: f32x8, y: f32x8) -> (f32x8, f32x8) {
    ((-(2.0 / PI) * (1.0 / tan_theta) * (1.0 / y.tan())).exp(),
     (-(1.0 / PI) * pow2(1.0 / tan_theta) * pow2(1.0 / y.tan())).exp())
}

fn gh(x: f32x8, tan_theta: f32x8, y: f32x8) -> f32x8 {
    let e = e12(tan_theta, y);

    x * (y.cos() + (y.sin() * tan_theta * (e.1 / (2.0 - e.0)))).abs()
}

fn compute_H(x: f32x8, ssa: f32x8) -> f32x8 {
    let y = (1.0 - ssa).sqrt();

    let r0 = (1.0 - y) / (1.0 + y);

    let Hinv = 1.0 - (1.0 - y) * x * (r0 + (1.0 - 0.5 * r0 - r0 * x) * ((1.0 + x) / x).ln());
    1.0 / Hinv
}

fn compute_Bc(g: f32x8, hc: f32x8) -> f32x8 {
    let zeta = (g / 2.0).tan().ln() / hc;

    let result = (1.0 + (1.0 - (-zeta).exp()) / zeta) / (2.0 * pow2(1.0 + zeta));
    
    // if result.is_nan() {
    //     1.0
    // } else {
    //     result
    // }
    
    result
}