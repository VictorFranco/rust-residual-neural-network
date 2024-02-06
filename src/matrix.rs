use std::ops::Add;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub value: Vec<Vec<f32>>
}

impl Matrix {

    #[allow(dead_code)]
    pub fn size(&self) {
        return println!("[ {}, {} ]", self.value.len(), self.value[0].len());
    }

    #[allow(dead_code)]
    pub fn transpose(&self) -> Matrix {
        let mut value = vec![vec![0.0; self.value.len()]; self.value[0].len()];
        for i in 0..self.value.len() {
            for j in 0..self.value[0].len() {
                value[j][i] = self.value[i][j];
            }
        }
        return Matrix { value };
    }

    pub fn dot(m_a: &Matrix, m_b: &Matrix) -> Matrix {
        if m_a.value[0].len() != m_b.value.len() {
            panic!("Dot product error");
        }
        let mut value = vec![];
        for i in 0..m_a.value.len() {
            let mut row = vec![];
            for j in 0..m_b.value[0].len() {
                let mut sum = 0.0;
                for k in 0..m_a.value[0].len() {
                    sum += m_a.value[i][k] * m_b.value[k][j];
                }
                row.push(sum);
            }
            value.push(row);
        }
        return Matrix { value };
    }

}

impl Add for Matrix {

    type Output = Matrix;

    fn add(self, m_b: Matrix) -> Matrix {
        if self.value.len() != m_b.value.len() || self.value[0].len() != m_b.value[0].len() {
            panic!("Addition error");
        }
        let mut value = vec![];
        for i in 0..self.value.len() {
            let mut row = vec![];
            for j in 0..m_b.value[0].len() {
                row.push(self.value[i][j] + m_b.value[i][j]);
            }
            value.push(row);
        }
        return Matrix { value };
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
