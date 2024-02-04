use crate::tensor::Tensor as Tensor;
use crate::matrix::Matrix as Matrix;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct ResNet {
    pub inputs: Tensor,
    pub labels: Tensor,
    pub rand: rand::rngs::StdRng,
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
    pub w3: Matrix,
    pub b3: Matrix,
    pub w4: Matrix,
    pub b4: Matrix,
}

impl ResNet {

    pub fn create_resnet(inputs: &Tensor, labels: &Tensor, seed: u64) -> ResNet {
        let inputs = inputs.clone();
        let labels = labels.clone();
        let mut rand = rand::rngs::StdRng::seed_from_u64(seed);
        let w1 = Matrix::random_matrix(4,4,&mut rand);
        let b1 = Matrix::random_matrix(1,4,&mut rand);
        let w2 = Matrix::random_matrix(4,4,&mut rand);
        let b2 = Matrix::random_matrix(1,4,&mut rand);
        let w3 = Matrix::random_matrix(4,4,&mut rand);
        let b3 = Matrix::random_matrix(1,4,&mut rand);
        let w4 = Matrix::random_matrix(4,4,&mut rand);
        let b4 = Matrix::random_matrix(1,4,&mut rand);
        ResNet { inputs, labels, rand, w1, b1, w2, b2, w3, b3, w4, b4 }
    }

    pub fn activation(matrix: Matrix) -> Matrix {
        let e = std::f32::consts::E;
        let sigmoid = |&x:&f32| 1.0 / (1.0 + e.powf(-x));
        let value = matrix.value.iter().map(|r| r.iter().map(sigmoid).collect()).collect();
        return Matrix { value };
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        let out1 = Self::activation(
            Matrix::add(&Matrix::dot(&input, &self.w1), &self.b1)
        );
        let out2 = Self::activation(
            Matrix::add(&Matrix::add(&Matrix::dot(&out1, &self.w2), &self.inputs.value[0]), &self.b2)
        );
        let out3 = Self::activation(
            Matrix::add(&Matrix::add(&Matrix::dot(&out2, &self.w3), &out1), &self.b3)
        );
        let out4 = Self::activation(
            Matrix::add(&Matrix::add(&Matrix::dot(&out3, &self.w4), &out2), &self.b4)
        );
        out4
    }

}
