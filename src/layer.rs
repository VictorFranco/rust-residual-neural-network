use crate::matrix::Matrix as Matrix;
use rand::{Rng, rngs::StdRng};

#[derive(Debug, Clone)]
pub struct Layer {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Matrix,
    pub biases: Matrix
}

impl Layer {

    pub fn new(in_features: usize, out_features: usize, rand: &mut StdRng) -> Layer {
        Layer {
            in_features,
            out_features,
            weights: Self::random_matrix(in_features, out_features, rand),
            biases: Self::random_matrix(1, out_features, rand)
        }
    }

    pub fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix {
        Matrix {
            value: (0..rows).map(|_| (0..cols).map(|_| rng.gen_range(-0.5..=0.5)).collect()).collect()
        }
    }

}

