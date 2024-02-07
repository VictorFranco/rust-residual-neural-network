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
                let mut equal = true;
                for k in 0..labels.value[i].value[j].len() {
                    if labels.value[i].value[j][k] != out.value[j][k] {
                        equal = false;
                    }
                }
                if equal {
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
            let input_size = self.inputs.value.len();
            let Layer { in_features, out_features, .. } = self.layers[0];
            let mut d_weights = vec![Matrix::new(in_features, out_features, 0.0); self.layers.len()];
            let mut d_biases = vec![Matrix::new(1, out_features, 0.0); self.layers.len()];
            for index in 0..input_size {
                let input = self.inputs.value[index].clone();
                let mut outs: Vec<Matrix> = vec![];
                let mut logits: Vec<Matrix> = vec![];
                let mut delta = Matrix::new(1, 4, 0.0);

                // forward
                for layer in 0..self.layers.len() {
                    let input_layer = if layer == 0 { input.clone() } else { outs.last().unwrap().clone() };
                    logits.push(Matrix::dot(&input_layer, &self.layers[layer].weights) + self.layers[layer].biases.clone());
                    // add residual connections
                    logits[layer] = match layer {
                        0 => logits[layer].clone(),
                        1 => logits[layer].clone() + input.clone(),
                        _ => logits[layer].clone() + outs[(layer as i32 - 2) as usize].clone()
                    };
                    outs.push(Self::sigmoid(&logits[layer]));
                }

                // backpropagation
                for layer in (0..self.layers.len()).rev() {
                    delta = match layer {
                        n if n == self.layers.len() - 1 => { // last layer
                            let error = self.labels.value[index].clone() - outs.last().unwrap().clone();
                            error * Self::sigmoid_prime(&logits[layer])
                        },  // hidden layers
                        _ => Matrix::dot(&delta, &self.layers[(layer as i32 + 1) as usize].weights.transpose()) * Self::sigmoid_prime(&logits[layer])
                    };
                    let input_layer = if layer == 0 { input.clone() } else { outs[(layer as i32) as usize - 1].clone() };
                    d_weights[layer] += Matrix::dot(&input_layer.transpose(), &delta);
                    d_biases[layer] += delta.clone();
                }
            }

            // update parameters
            for layer in 0..self.layers.len() {
                d_weights[layer] = Matrix::scalar_mul(lr / input_size as f32, d_weights[layer].clone());
                d_biases[layer] = Matrix::scalar_mul(lr / input_size as f32, d_biases[layer].clone());
                self.layers[layer].weights += d_weights[layer].clone();
                self.layers[layer].biases += d_biases[layer].clone();
            }

            // testing
            if epoch % 1 == 0 {
                let value = self.inputs.value.iter().map(|x| self.forward(x)).collect();
                let outputs = Tensor { value };
                let loss = Self::mse(&self.labels, &outputs);
                let acc = Self::accuracy(&self.labels, &outputs);
                println!("epoch: {}/{} loss: {} acc: {}", epoch + 1, epochs, loss, acc);
            }
        }
    }

}
