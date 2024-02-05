use crate::matrix::Matrix as Matrix;
use rand::{Rng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Matrix,
    pub biases: Matrix
}

impl Layer {

    pub fn new(in_channel: i32, out_channel: i32, rand: &mut StdRng) -> Layer {
        Layer {
            weights: Self::random_matrix(in_channel, out_channel, rand),
            biases: Self::random_matrix(1, out_channel, rand)
        }
    }

    pub fn random_matrix(rows: i32, cols: i32, rng: &mut StdRng) -> Matrix {
        Matrix {
            value: (0..rows).map(|_| (0..cols).map(|_| rng.gen_range(-0.5..=0.5)).collect()).collect()
        }
    }

}

