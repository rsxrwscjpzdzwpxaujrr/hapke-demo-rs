use std::cell::UnsafeCell;
use std::ops::Deref;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Default)]
pub(crate) struct DoubleBuffer<T> {
    buffers: [UnsafeCell<T>; 2],
    current: AtomicU8,
}

impl<T: Clone> From<T> for DoubleBuffer<T> {
    fn from(value: T) -> Self {
        Self {
            buffers: [UnsafeCell::new(value.clone()), UnsafeCell::new(value)],
            current: Default::default(),
        }
    }
}

impl<T> DoubleBuffer<T> {
    pub(crate) fn flip(&self) {
        self.current.store(1 - self.current.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub(crate) fn get_mut(&self) -> &mut T {
        unsafe {
            &mut *self.buffers[self.current.load(Ordering::Relaxed) as usize].get()
        }
    }
}

impl<T> Deref for DoubleBuffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe {
            &*self.buffers[(1 - self.current.load(Ordering::Relaxed)) as usize].get()
        }
    }
}

unsafe impl<T> Send for DoubleBuffer<T> {}
unsafe impl<T> Sync for DoubleBuffer<T> {}