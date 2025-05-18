use std::sync::RwLock;
use std::time::{Duration, Instant};

pub(crate) struct Averager {
    interval: Duration,
    last_time: RwLock<Instant>,
    measurements: RwLock<Vec<f32>>,
    value: RwLock<f32>,
}

impl Averager {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_time: RwLock::new(Instant::now()),
            measurements: Default::default(),
            value: Default::default(),
        }
    }

    pub fn add_measurement(&self, measurement: f32) {
        if self.last_time.read().unwrap().elapsed() >= self.interval {
            {
                let measurements = self.measurements.read().unwrap();

                *self.value.write().unwrap() = measurements.iter().sum::<f32>()
                    / measurements.len() as f32;
            }

            self.measurements.write().unwrap().clear();

            *self.last_time.write().unwrap() = Instant::now();
        }

        self.measurements.write().unwrap().push(measurement);
    }

    pub fn average(&self) -> f32 {
        *self.value.read().unwrap()
    }
}

