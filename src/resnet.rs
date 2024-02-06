use crate::matrix::Matrix as Matrix;
use crate::tensor::Tensor as Tensor;
use crate::layer::Layer as Layer;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct ResNet {
    pub inputs: Tensor,
    pub labels: Tensor,
    pub rand: rand::rngs::StdRng,
    pub layers: Vec<Layer>
}

impl ResNet {

    pub fn new(inputs: &Tensor, labels: &Tensor, seed: u64) -> ResNet {
        let inputs = inputs.clone();
        let labels = labels.clone();
        let mut rand = rand::rngs::StdRng::seed_from_u64(seed);
        let layers = vec![
            Layer::new(4, 4, &mut rand),
            Layer::new(4, 4, &mut rand),
            Layer::new(4, 4, &mut rand),
            Layer::new(4, 4, &mut rand)
        ];
        ResNet { inputs, labels, rand, layers }
    }

    pub fn threshold(matrix: &Matrix) -> Matrix {
        let threshold = |&x: &f32| if x > 0.5 { 1.0 } else { 0.0 };
        let value = matrix.value.iter().map(|r| r.iter().map(threshold).collect()).collect();
        Matrix { value }
    }

    pub fn sigmoid(matrix: &Matrix) -> Matrix {
        let e = std::f32::consts::E;
        let sigmoid = |&x: &f32| 1.0 / (1.0 + e.powf(-x));
        let value = matrix.value.iter().map(|r| r.iter().map(sigmoid).collect()).collect();
        Matrix { value }
    }

    pub fn sigmoid_prime(logit: &Matrix) -> Matrix {
        let sigmoid = Self::sigmoid(&logit);
        sigmoid.clone() * (Matrix::new(logit.value.len(), logit.value[0].len(), 1.0) - sigmoid.clone())
    }

    pub fn mse(labels: &Tensor, outputs: &Tensor) -> f32 {
        let mut sum = 0.0;
        for i in 0..labels.value.len() {
            for j in 0..labels.value[i].value.len() {
                for k in 0..labels.value[i].value[j].len() {
                    sum += (labels.value[i].value[j][k] - outputs.value[i].value[j][k]).powf(2.0);
                }
            }
        }
        return 1.0 / labels.value.len() as f32 * sum;
    }

    pub fn accuracy(labels: &Tensor, outputs: &Tensor) -> f32 {
        let mut sum = 0.0;
        for i in 0..labels.value.len() {
            let out = Self::threshold(&outputs.value[i]);
            for j in 0..labels.value[i].value.len() {
                let mut flag = true;
                for k in 0..labels.value[i].value[j].len() {
                    if labels.value[i].value[j][k] != out.value[j][k] {
                        flag = false;
                    }
                }
                if flag {
                    sum = sum + 1.0;
                }
            }
        }
        sum / labels.value.len() as f32
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let input = input.clone();
        let mut outs: Vec<Matrix> = vec![];
        for index in 0..self.layers.len() {
            let input_layer = if index == 0 { input.clone() } else { outs.last().unwrap().clone() };
            let logit = Matrix::dot(&input_layer, &self.layers[index].weights) + self.layers[index].biases.clone();
            // add residual connections
            let logit = match index {
                0 => logit.clone(),
                1 => logit.clone() + input.clone(),
                _ => logit.clone() + outs[(index as i32 - 2) as usize].clone()
            };
            outs.push(Self::sigmoid(&logit));
        }
        outs.last().unwrap().clone()
    }

    pub fn train(&mut self, lr: f32, epochs: usize) {
        for epoch in 0..epochs {
            let mut d_weights = vec![Matrix::new(4, 4, 0.0); 4];
            let mut d_biases = vec![Matrix::new(1, 4, 0.0); 4];
            for index in 0..self.inputs.value.len() {
                let input = self.inputs.value[index].clone();
                let mut outs: Vec<Matrix> = vec![Matrix::new(1, 4, 0.0);4];
                let mut logits: Vec<Matrix> = vec![Matrix::new(1, 4, 0.0);4];

                // forward
                logits[0] = Matrix::dot(&input.clone(), &self.layers[0].weights) + self.layers[0].biases.clone();
                outs[0] = Self::sigmoid(&logits[0]);
                logits[1] = Matrix::dot(&outs[0], &self.layers[1].weights) + self.layers[1].biases.clone() + input.clone();
                outs[1] = Self::sigmoid(&logits[1]);
                logits[2] = Matrix::dot(&outs[1], &self.layers[2].weights) + self.layers[2].biases.clone() + outs[0].clone();
                outs[2] = Self::sigmoid(&logits[2]);
                logits[3] = Matrix::dot(&outs[2], &self.layers[3].weights) + self.layers[3].biases.clone() + outs[1].clone();
                outs[3] = Self::sigmoid(&logits[3]);

                // backpropagation
                let error = self.labels.value[index].clone() - outs.last().unwrap().clone();
                let delta = error * Self::sigmoid_prime(&logits[3]);
                d_weights[3] = d_weights[3].clone() + Matrix::dot(&outs[2].transpose(), &delta);
                d_biases[3] = d_biases[3].clone() + delta.clone();

                let delta = Matrix::dot(&delta, &self.layers[3].weights.transpose()) * Self::sigmoid_prime(&logits[2]);
                d_weights[2] = d_weights[2].clone() + Matrix::dot(&outs[1].transpose(), &delta);
                d_biases[2] = d_biases[2].clone() + delta.clone();

                let delta = Matrix::dot(&delta, &self.layers[2].weights.transpose()) * Self::sigmoid_prime(&logits[1]);
                d_weights[1] = d_weights[1].clone() + Matrix::dot(&outs[0].transpose(), &delta);
                d_biases[1] = d_biases[1].clone() + delta.clone();

                let delta = Matrix::dot(&delta, &self.layers[1].weights.transpose()) * Self::sigmoid_prime(&logits[0]);
                d_weights[0] = d_weights[0].clone() + Matrix::dot(&input.transpose(), &delta);
                d_biases[0] = d_biases[0].clone() + delta.clone();

            }

            // update parameters
            for layer in 0..self.layers.len() {
                self.layers[layer].weights = self.layers[layer].weights.clone() + Matrix::scalar_mul(lr / 16.0, d_weights[layer].clone());
                self.layers[layer].biases = self.layers[layer].biases.clone() + Matrix::scalar_mul(lr / 16.0, d_biases[layer].clone());
            }

            // testing
            if epoch % 10 == 0 {
                let value = self.inputs.value.iter().map(|x| self.forward(x)).collect();
                let outputs = Tensor { value };
                let loss = Self::mse(&self.labels, &outputs);
                let acc = Self::accuracy(&self.labels, &outputs);
                println!("epoch: {}/{} loss: {} acc: {}", epoch + 1, epochs, loss, acc);
            }
        }
    }

}
