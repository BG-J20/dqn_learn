use crate::network::Network;
use crate::replay_buffer::{ReplayBuffer, Experience};

use rand::Rng;

pub struct Agent {
    pub network: Network,
    pub replay_buffer: ReplayBuffer,
    pub epsilon: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub gamma: f32,
    pub action_space: usize,
}

impl Agent {
    pub fn new(network: Network, replay_buffer: ReplayBuffer, action_space: usize) -> Self {
        Self {
            network,
            replay_buffer,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.1,
            action_space,
        }
    }
}