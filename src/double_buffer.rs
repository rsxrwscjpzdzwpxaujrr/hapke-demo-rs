use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Default)]
pub(crate) struct DoubleBuffer<T> {
    buffers: [RwLock<T>; 2],
    current: AtomicU8,
}

impl<T: Clone> From<T> for DoubleBuffer<T> {
    fn from(value: T) -> Self {
        Self {
            buffers: [RwLock::new(value.clone()), RwLock::new(value)],
            current: Default::default(),
        }
    }
}

impl<T> DoubleBuffer<T> {
    pub(crate) fn flip(&self) {
        self.current.store(1 - self.current.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub(crate) fn get_mut(&self) -> RwLockWriteGuard<'_, T> {
        self.buffers[self.current.load(Ordering::Relaxed) as usize].write().unwrap()
    }

    pub(crate) fn read(&self) -> Option<RwLockReadGuard<'_, T>> {
        self.buffers[1 - self.current.load(Ordering::Relaxed) as usize].try_read().ok()
    }
}
