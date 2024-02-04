mod matrix;
mod tensor;

use matrix::Matrix as Matrix;
use tensor::Tensor as Tensor;
use rand::SeedableRng;

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
