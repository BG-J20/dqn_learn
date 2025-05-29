pub struct Env {
    pub state_size: usize,
    pub action_size: usize,
    state: Vec<f32>,
    pub done: bool,
}

impl Env {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        Env {
            state_size,
            action_size,
            state: vec![0.0; state_size],
            done: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn reset(&mut self) -> Vec<f32> {
        self.state = vec![0.0; self.state_size];
        self.state.clone()
    }

    pub fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        //фиктивня логика среды. сдвигаем состояние и вовзвращаем случайное вознаграждение

        for val in &mut self.state {
            *val += action as f32 * 0.01;
        }

        let reward = if action == 1 {1.0} else {0.0}; // примитивное вознагражение от чата гпт
        let done = self.state.iter().any(|&x| x > 1.0); // завершение епизода от чата гпт

        (self.state.clone(), reward, done)
    }
}