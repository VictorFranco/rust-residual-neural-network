use crate::matrix::Matrix as Matrix;

pub struct Tensor {
    pub value: Vec<Matrix>
}

impl Tensor {

    pub fn vec2matrices(vector: Vec<Vec<Vec<f32>>>) -> Vec<Matrix> {
        vector.iter().map(|value| Matrix { value: value.to_vec() } ).collect()
    }

}

impl std::fmt::Display for Tensor {

    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for matrix in &self.value {
            print!("{}", matrix);
        }
        Ok(())
    }

}
