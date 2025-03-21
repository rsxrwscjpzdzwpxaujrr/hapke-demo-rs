use std::f32::consts::PI;
use crate::shader::Shader;
use crate::vec3::Vec3;

#[derive(Copy, Clone, Default)]
pub(crate) struct HapkeParams {
    pub w    : f32, // Single scattering albedo
    pub b    : f32, // Henyey-Greenstein double-lobed single particle phase function parameter
    pub c    : f32, // Henyey-Greenstein double-lobed single particle phase function parameter
    pub Bc0  : f32, // Amplitude of Coherent Backscatter Opposition Effect (CBOE) - fixed at 0.0
    pub hc   : f32, // Angular width of CBOE - fixed at 1.0
    pub Bs0  : f32, // Amplitude of Shadow Hiding Opposition Effect (SHOE)
    pub hs   : f32, // Angular width of SHOE
    pub theta: f32, // Effective value of the photometric roughness - fixed at 23.657
    pub phi  : f32, // Filling factor - fixed at 1.0
}

pub(crate) struct Hapke {}

fn acos_clamped(x: f32) -> f32 {
    x.clamp(-1.0, 1.0).acos()
}

impl Shader<HapkeParams> for Hapke {
    fn brdf(&self, light: &Vec3<f32>, normal: &Vec3<f32>, camera: &Vec3<f32>, data: &HapkeParams) -> f32 {
        //let K = -f32::ln(1.0 - (1.209 * data.phi.powf(2.0 / 3.0))) / data.phi.powf(2.0 / 3.0);

        if data.w == 0.0 {
            return 0.0;
        }

        let mu = -camera.dot(normal);
        let mu0 = -light.dot(normal);

        if mu <= 0.0 || mu0 <= 0.0 {
            return 0.0;
        }

        // let tan_theta = data.theta.to_radians().tan();
        let tan_theta = 0.0f32.to_radians().tan();

        let K = 1.0 - data.phi;

        let i = -acos_clamped(light.dot(normal));
        let e = -acos_clamped(camera.dot(normal));
        let g = acos_clamped(camera.dot(light));

        let Ei = e12(tan_theta, i);
        let Ee = e12(tan_theta, e);

        let phi = acos_clamped((camera.dot(light) - (mu * mu0)) / (f32::sin(e) * f32::sin(i)));
        //let phi = phi.clamp(0.0, FRAC_PI_2);

        //let phi = if phi.is_nan() { PI } else { phi };

        let temp = if i <= e {(
            ((phi.cos() * Ee.1) + (phi / 2.0).sin().powi(2) * Ei.1) / (2.0 - Ee.0 - ((phi / PI) * Ei.0)),
            (Ee.1               - (phi / 2.0).sin().powi(2) * Ei.1) / (2.0 - Ee.0 - ((phi / PI) * Ei.0)),
        )}
        else {(
            (Ei.1               - (phi / 2.0).sin().powi(2) * Ee.1) / (2.0 - Ei.0 - ((phi / PI) * Ee.0)),
            ((phi.cos() * Ei.1) + (phi / 2.0).sin().powi(2) * Ee.1) / (2.0 - Ei.0 - ((phi / PI) * Ee.0)),
        )};

        let x = 1.0 / (1.0 + (PI * tan_theta.powi(2))).sqrt();

        let mu0_e = x * (mu0 + (i.sin() * tan_theta * temp.0));
        let mu_e  = x * (mu  + (e.sin() * tan_theta * temp.1));
        
        // let mu0_e = mu0;
        // let mu_e = mu;

        let ls = mu0_e / (mu_e + mu0_e);

        let p =
            ((1.0 + data.c) / 2.0) *
                (1.0 - (data.b * data.b)) / (1.0 - (2.0 * data.b * f32::cos(g)) + (data.b).powi(2)).powf(3.0 / 2.0) +
                ((1.0 - data.c) / 2.0) *
                    (1.0 - (data.b * data.b)) / (1.0 + (2.0 * data.b * f32::cos(g)) + (data.b).powi(2)).powf(3.0 / 2.0);

        let bs = 1.0 / (1.0 + (f32::tan(g / 2.0) / data.hs));

        let M = (compute_H(mu0_e / K, data.w) * compute_H(mu_e / K, data.w)) - 1.0;

        let f = f32::exp(-2.0 * f32::tan(phi / 2.0));
        
        let f = if f == f32::INFINITY { 1.0 } else if f == f32::NEG_INFINITY { 0.0 } else {f};

        let ghi = gh(x, tan_theta, i);
        let ghe = gh(x, tan_theta, e);

        let temp = if i <= e {
            mu0 / ghi
        }
        else {
            mu / ghe
        };

        let shadowing = (mu_e / ghe) * (mu0 / ghi) * (x / (1.0 - f + (f * x * temp)));

        // let shadowing = if shadowing.is_normal() { shadowing } else { 1.0 };

        let result = ls * K * (data.w / 4.0) * (p * (1.0 + data.Bs0 * bs) + M) * (1.0 + data.Bc0 * compute_Bc(g, data.hc)) * shadowing;

        result
    }
}

fn e12(tan_theta: f32, y: f32) -> (f32, f32) {
    (f32::exp((-(2.0 / PI) * (1.0 / tan_theta) * (1.0 / f32::tan(y)))),
     f32::exp((-(1.0 / PI) * f32::powf(1.0 / tan_theta, 2.0) * f32::powf(1.0 / f32::tan(y), 2.0))))
}

fn gh(x: f32, tan_theta: f32, y: f32) -> f32 {
    let e = e12(tan_theta, y);

    x * (f32::cos(y) + (f32::sin(y) * tan_theta * (e.1 / (2.0 - e.0)))).abs()
}

fn compute_H(x: f32, ssa: f32) -> f32 {
    let y = f32::sqrt(1.0 - ssa);

    let r0 = (1.0 - y) / (1.0 + y);

    let Hinv = 1.0 - (1.0 - y) * x * (r0 + (1.0 - 0.5 * r0 - r0 * x) * f32::ln((1.0 + x) / x));
    1.0 / Hinv
}

fn compute_Bc(g: f32, hc: f32) -> f32 {
    let zeta = f32::ln(f32::tan(g / 2.0)) / hc;

    let result = (1.0 + (1.0 - f32::exp(-zeta)) / zeta) / (2.0 * f32::powi(1.0 + zeta, 2));
    
    if result.is_nan() {
        1.0
    } else {
        result
    }
}