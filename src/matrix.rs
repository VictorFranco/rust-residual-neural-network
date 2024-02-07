use std::ops::{Add, AddAssign, Sub, Mul};

#[derive(Debug, Clone)]
pub struct Matrix {
    pub value: Vec<Vec<f32>>
}

impl Matrix {

    pub fn new(rows: usize, cols: usize, value: f32) -> Matrix {
        Matrix {
            value: vec![vec![value; cols]; rows]
        }
    }

    #[allow(dead_code)]
    pub fn size(&self) {
        return println!("[ {}, {} ]", self.value.len(), self.value[0].len());
    }

    #[allow(dead_code)]
    pub fn transpose(&self) -> Matrix {
        let value = (0..self.value[0].len())
            .map(|i| (0..self.value.len())
            .map(|j| self.value[j][i]).collect()).collect();
        Matrix { value }
    }

    pub fn scalar_mul(scalar: f32, matrix: Matrix) -> Matrix {
        if matrix.value.len() == 0 || matrix.value[0].len() == 0 {
            panic!("Addition error");
        }
        let value = (0..matrix.value.len())
            .map(|i| (0..matrix.value[i].len())
            .map(|j| scalar * matrix.value[i][j]).collect()).collect();
        Matrix { value }
    }

    pub fn dot(m_a: &Matrix, m_b: &Matrix) -> Matrix {
        if m_a.value[0].len() != m_b.value.len() {
            panic!("Dot product error");
        }
        let value = (0..m_a.value.len())
            .map(|i| (0..m_b.value[0].len())
            .map(|j| (0..m_a.value[0].len())
            .map(|k| m_a.value[i][k] * m_b.value[k][j]).sum()).collect()).collect();
        Matrix { value }
    }

}

impl Add for Matrix {

    type Output = Matrix;

    fn add(self, matrix: Matrix) -> Matrix {
        if self.value.len() != matrix.value.len() || self.value[0].len() != matrix.value[0].len() {
            panic!("Addition error");
        }
        let value = (0..self.value.len())
            .map(|i| (0..self.value[i].len())
            .map(|j| self.value[i][j] + matrix.value[i][j]).collect()).collect();
        Matrix { value }
    }

}

impl AddAssign for Matrix {

    fn add_assign(&mut self, matrix: Matrix) {
        if self.value.len() != matrix.value.len() || self.value[0].len() != matrix.value[0].len() {
            panic!("Addition error");
        }
        let value = (0..self.value.len())
            .map(|i| (0..self.value[i].len())
            .map(|j| self.value[i][j] + matrix.value[i][j]).collect()).collect();
        *self = Matrix { value };
    }

}

impl Sub for Matrix {

    type Output = Matrix;

    fn sub(self, matrix: Matrix) -> Matrix {
        if self.value.len() != matrix.value.len() || self.value[0].len() != matrix.value[0].len() {
            panic!("Subtraction error");
        }
        let value = (0..self.value.len())
            .map(|i| (0..self.value[i].len())
            .map(|j| self.value[i][j] - matrix.value[i][j]).collect()).collect();
        Matrix { value }
    }

}

impl Mul for Matrix {

    type Output = Matrix;

    fn mul(self, matrix: Matrix) -> Matrix { // Hadamard product
        if self.value.len() != matrix.value.len() || self.value[0].len() != matrix.value[0].len() {
            panic!("Multiplication error");
        }
        let value = (0..self.value.len())
            .map(|i| (0..self.value[i].len())
            .map(|j| self.value[i][j] * matrix.value[i][j]).collect()).collect();
        Matrix { value }
    }

}

impl std::fmt::Display for Matrix {

    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.value {
            print!("[ ");
            for (i, elm) in row.iter().enumerate() {
                print!("{}", elm);
                if i  < row.len() - 1 {
                    print!(", ");
                }
            }
            println!(" ]");
        }
        Ok(())
    }

}
