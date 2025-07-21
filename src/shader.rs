use std::array;
use std::cell::Cell;
use wide::f32x8;
use crate::SIMD_SIZE;
use crate::vec3::Vec3;

pub(crate) struct ValueDebugger {
    string: Cell<String>,
    empty: Cell<bool>,
}

impl Default for ValueDebugger {
    fn default() -> Self {
        Self {
            string: Default::default(),
            empty: Cell::new(true),
        }
    }
}

impl ValueDebugger {
    pub(crate) fn assign_str(&self, string: String) {
        self.string.set(string);
        self.empty.set(false);
    }

    pub(crate) fn get(self) -> String {
        self.string.into_inner()
    }

    pub(crate) fn empty(&self) -> bool {
        self.empty.get()
    }
}

pub(crate) trait Shader<T, const CHANNELS: usize> {
    fn new(params: [T; CHANNELS]) -> Self;
    fn brdf(
        &self,
        light: &Vec3<f32x8>,
        normal: &Vec3<f32x8>,
        camera: &Vec3<f32x8>,
        debugger: [Option<&ValueDebugger>; SIMD_SIZE]
    ) -> [f32x8; CHANNELS];

    fn brdf_non_simd(
        &self,
        light: &Vec3<f32>,
        normal: &Vec3<f32>,
        camera: &Vec3<f32>,
        debugger: Option<&ValueDebugger>
    ) -> [f32; CHANNELS] {
        self.brdf(
            &[light.x.into(), light.y.into(), light.z.into()].into(),
            &[normal.x.into(), normal.y.into(), normal.z.into()].into(),
            &[camera.x.into(), camera.y.into(), camera.z.into()].into(),
            [debugger, None, None, None, None, None, None, None],
        ).map(|value| value.as_array_ref()[0])
    }
}
