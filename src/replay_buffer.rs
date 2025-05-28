use rand::seq::SliceRandom;
use rand::prelude::IndexedRandom;

pub struct Experience {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, exp: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(exp);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Experience> {
        let mut rng = rand::thread_rng();
        self.buffer
            .choose_multiple(&mut rng, batch_size)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()

    }
}