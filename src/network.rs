use rand::Rng;
use serde::{Serialize, Deserialize};

/// Один слой нейросети (полносвязный)
#[derive(Serialize, Deserialize, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Layer {
    /// Создание слоя: веса инициализируются случайно в диапазоне [-1.0, 1.0]
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0)) // случайный вес
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self { weights, biases }
    }

    /// Прямой проход через слой: input → ReLU(weights * input + bias)
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| {
                let sum: f32 = w_row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                (sum + b).max(0.0) // ReLU
            })
            .collect()
    }

    /// Обновление весов слоя (грубое приближение обучения)
    pub fn update(&mut self, input: &Vec<f32>, output: &Vec<f32>, target: &Vec<f32>, learning_rate: f32) {
        for i in 0..self.weights.len() {
            let error = target[i] - output[i]; // ошибка
            for j in 0..self.weights[i].len() {
                self.weights[i][j] += learning_rate * error * input[j];
            }
            self.biases[i] += learning_rate * error;
        }
    }
}

/// Нейросеть, состоящая из нескольких слоёв


impl Network {
    /// Создание сети по слоям: [вход, скрытые…, выход]
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();

        Self { layers }
    }

    /// Прямой проход по всей сети
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }

    /// Простая функция обучения (только выходной слой): состояние + целевые значения
    pub fn train(&mut self, input: Vec<f32>, target: Vec<f32>, learning_rate: f32) {
        // forward с запоминанием промежуточных результатов
        let mut activations = vec![input.clone()];
        let mut current = input;

        for layer in &self.layers {
            current = layer.forward(&current);
            activations.push(current.clone());
        }

        // обновим только последний слой (грубо, без backpropagation)
        if let Some(last_layer) = self.layers.last_mut() {
            let last_input = &activations[activations.len() - 2];
            let output = &activations[activations.len() - 1];
            last_layer.update(last_input, output, &target, learning_rate);
        }
    }
}
