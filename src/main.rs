use rand::{SeedableRng, Rng};

#[derive(Debug)]
struct Tensor {
    value: Vec<Matrix>
}

impl Tensor {

    fn vec2matrices(vector: Vec<Vec<Vec<f32>>>) -> Vec<Matrix> {
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

#[derive(Debug)]
struct Matrix {
    value: Vec<Vec<f32>>
}

impl Matrix {

    fn random_matrix(rows: i32, cols: i32, rng: &mut rand::rngs::StdRng) -> Matrix {
        Matrix {
            value: (0..rows).map(|_| (0..cols).map(|_| rng.gen()).collect()).collect()
        }
    }

    #[allow(dead_code)]
    fn size(&self) {
        return println!("[ {}, {} ]", self.value.len(), self.value[0].len());
    }

    #[allow(dead_code)]
    fn transpose(&self) -> Matrix {
        let mut value = vec![vec![0.0; self.value.len()]; self.value[0].len()];
        for i in 0..self.value.len() {
            for j in 0..self.value[0].len() {
                value[j][i] = self.value[i][j];
            }
        }
        return Matrix { value };
    }

    fn dot(m_a: &Matrix, m_b: &Matrix) -> Matrix {
        if m_a.value[0].len() != m_b.value.len() {
            panic!("Dot product error");
        }
        let mut value = vec![];
        for i in 0..m_a.value.len() {
            let mut row = vec![];
            for j in 0..m_b.value[0].len() {
                let mut sum = 0.0;
                for k in 0..m_a.value[0].len() {
                    sum += m_a.value[i][k] + m_b.value[j][k];
                }
                row.push(sum);
            }
            value.push(row);
        }
        return Matrix { value };
    }

    fn add(m_a: &Matrix, m_b: &Matrix) -> Matrix {
        if m_a.value.len() != m_b.value.len() || m_a.value[0].len() != m_b.value[0].len() {
            panic!("Addition error");
        }
        let mut value = vec![];
        for i in 0..m_a.value.len() {
            let mut row = vec![];
            for j in 0..m_b.value[0].len() {
                row.push(m_a.value[i][j] + m_b.value[i][j]);
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

fn activation(matrix: Matrix) -> Matrix {
    let e = std::f32::consts::E;
    let sigmoid = |&x:&f32| 1.0 / (1.0 + e.powf(-x));
    let value = matrix.value.iter().map(|r| r.iter().map(sigmoid).collect()).collect();
    return Matrix { value };
}

fn main() {

    let array2vec3d = |m: &[[i32; 4];1]| m.iter().map(|r| r.map(|e| e as f32).to_vec()).collect();

    let inputs = Tensor {
        value: Tensor::vec2matrices([
                   [[0, 0, 0, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 1]], [[0, 1, 0, 0]], [[0, 1, 1, 1]],
                   [[1, 0, 0, 0]], [[1, 0, 0, 1]], [[1, 0, 1, 1]], [[1, 1, 0, 0]], [[1, 1, 1, 1]]
               ].iter().map(array2vec3d).collect())
    };

    let _outputs = Tensor {
        value: Tensor::vec2matrices([
                   [[0, 0, 0, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 1]], [[0, 1, 0, 0]], [[0, 1, 1, 1]],
                   [[1, 0, 0, 0]], [[1, 0, 0, 1]], [[1, 0, 1, 1]], [[1, 1, 0, 0]], [[1, 1, 1, 1]]
               ].iter().map(array2vec3d).collect())
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(28);

    let w1 = Matrix::random_matrix(4,4,&mut rng);
    let b1 = Matrix::random_matrix(1,4,&mut rng);
    let w2 = Matrix::random_matrix(4,4,&mut rng);
    let b2 = Matrix::random_matrix(1,4,&mut rng);
    let w3 = Matrix::random_matrix(4,4,&mut rng);
    let b3 = Matrix::random_matrix(1,4,&mut rng);
    let w4 = Matrix::random_matrix(4,4,&mut rng);
    let b4 = Matrix::random_matrix(1,4,&mut rng);

    let out1 = activation(Matrix::add(&Matrix::dot(&inputs.value[0], &w1), &b1));
    let out2 = activation(Matrix::add(&Matrix::add(&Matrix::dot(&out1, &w2), &inputs.value[0]), &b2));
    let out3 = activation(Matrix::add(&Matrix::add(&Matrix::dot(&out2, &w3), &out1), &b3));
    let out4 = activation(Matrix::add(&Matrix::add(&Matrix::dot(&out3, &w4), &out2), &b4));

    println!("{}", out4);

}
