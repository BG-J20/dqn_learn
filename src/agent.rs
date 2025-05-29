use crate::network::Network;
use crate::replay_buffer::{ReplayBuffer, Experience};

use rand::{Rng, rng};
use rand::seq::SliceRandom;

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
            gamma: 0.99,
            replay_buffer,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.1,
            action_space,
        }
    }
/* Этап	Описание
1. sample()	Выбирает случайные Experience из буфера
2. target	Вычисляет целевое значение по формуле Беллмана
3. forward()	Получает текущие Q-значения
4. Обновление	Меняет Q[action] на target, остальные оставляет как есть
5. train()	Передаёт обновлённый вектор в сеть для обучения
6. epsilon decay	Делает агента со временем менее случайным

 */
    pub fn learn(&mut self, batch_size: usize) {
        let batch = self.replay_buffer.sample(batch_size);

        for experience in batch {
            let target = if experience.done {
                experience.reward
            } else {
                let next_q_values = self.network.forward(experience.next_state.clone());
                let max_next_q = next_q_values
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                experience.reward + self.gamma * max_next_q
            };

            let mut q_values = self.network.forward(experience.next_state.clone());
            q_values[experience.action] = target;

            self.network.train(experience.state.clone(), q_values, 0.01);
        }

        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }

    pub fn remember(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    pub fn select_action(&mut self, state: Vec<f32>) -> usize {
        let mut rng = rand::thread_rng();

        if rng.r#gen::<f32>() < self.epsilon {
            //случайной действие
            rng.r#gen_range(0..self.action_space)
        } else {
            //действия из сети
            let q_values = self.network.forward(state);
            argmax(&q_values)
        }
    }

    //добавление опыта в буфер
    pub fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.push(experience);
    }

    // Проверяет, достаточно ли опыта в буфере, чтобы начать обучение
    pub fn can_learn(&self, batch_size: usize) -> bool {
        self.replay_buffer.len() >= batch_size
    }

    //обновить е после каждой итерации
    pub fn decay_epsilon(&mut self) {
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }
}

//выбор индекса максимального значения
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

