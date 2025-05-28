use rand::Rng;

/// Один слой нейросети (полносвязный)
pub struct Layer {
    pub weights: Vec<Vec<f32>>, // матрица весов: [output_size][input_size]
    pub biases: Vec<f32>,       // вектор смещений: [output_size]
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

        let biases = vec![0.0; output_size]; // начальные смещения — 0

        Self { weights, biases }
    }

    /// Прямой проход через слой: input → ReLU(weights * input + bias)
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| {
                // Скалярное произведение весов на вход + bias
                let sum: f32 = w_row
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                // Активация ReLU: заменяет отрицательные значения на 0
                (sum + b).max(0.0)
            })
            .collect()
    }
}

/// Нейросеть, состоящая из нескольких слоёв
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// Создание сети по описанию размеров слоёв, например: [4, 16, 8, 2]
    /// Где 4 — вход, 2 — выход, между ними 2 скрытых слоя
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2) // парами [in, out]
            .map(|pair| Layer::new(pair[0], pair[1]))
            .collect();

        Self { layers }
    }

    /// Прямой проход по всем слоям сети
    pub fn forward(&self, mut input: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }
}
