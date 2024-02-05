use crate::tensor::Tensor as Tensor;
use crate::matrix::Matrix as Matrix;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct Layer {
    weights: Matrix,
    biases: Matrix
}

#[derive(Debug, Clone)]
pub struct ResNet {
    pub inputs: Tensor,
    pub labels: Tensor,
    pub rand: rand::rngs::StdRng,
    pub layers: Vec<Layer>
}

impl ResNet {

    pub fn create_resnet(inputs: &Tensor, labels: &Tensor, seed: u64) -> ResNet {
        let inputs = inputs.clone();
        let labels = labels.clone();
        let mut rand = rand::rngs::StdRng::seed_from_u64(seed);
        let layers = vec![
            Layer { weights: Matrix::random_matrix(4, 4, &mut rand), biases: Matrix::random_matrix(1, 4, &mut rand) },
            Layer { weights: Matrix::random_matrix(4, 4, &mut rand), biases: Matrix::random_matrix(1, 4, &mut rand) },
            Layer { weights: Matrix::random_matrix(4, 4, &mut rand), biases: Matrix::random_matrix(1, 4, &mut rand) },
            Layer { weights: Matrix::random_matrix(4, 4, &mut rand), biases: Matrix::random_matrix(1, 4, &mut rand) }
        ];
        ResNet { inputs, labels, rand, layers }
    }

    pub fn activation(matrix: Matrix) -> Matrix {
        let e = std::f32::consts::E;
        let sigmoid = |&x:&f32| 1.0 / (1.0 + e.powf(-x));
        let value = matrix.value.iter().map(|r| r.iter().map(sigmoid).collect()).collect();
        Matrix { value }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let input = input.clone();
        let layers = self.layers.clone();
        let out1 = Self::activation(Matrix::dot(&input, &layers[0].weights) + layers[0].biases.clone());
        let out2 = Self::activation(Matrix::dot(&out1, &layers[1].weights) + layers[1].biases.clone() + input);
        let out3 = Self::activation(Matrix::dot(&out2, &layers[2].weights) + layers[2].biases.clone() + out1);
        Self::activation(Matrix::dot(&out3, &layers[3].weights) + layers[3].biases.clone() + out2)
    }

}
