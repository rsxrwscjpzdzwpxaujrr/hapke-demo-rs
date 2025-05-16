use std::cell::Cell;
use wide::f32x8;
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

pub(crate) trait Shader<T> {
    fn brdf(
        &self, 
        light: &Vec3<f32x8>, 
        normal: &Vec3<f32x8>, 
        camera: &Vec3<f32x8>, 
        params: &T, 
        debugger: [Option<&ValueDebugger>; 8]
    ) -> f32x8;
}