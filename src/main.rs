mod matrix;
mod tensor;
mod layer;
mod resnet;

use tensor::Tensor as Tensor;
use resnet::ResNet as ResNet;

fn main() {

    // hyperparameters
    let seed = 74;
    let epochs = 11;
    let lr = 1.0;

    let array2vec3d = |m: &[[i32; 4];1]| m.iter().map(|r| r.map(|e| e as f32).to_vec()).collect();

    let inputs = Tensor {
        value: Tensor::vec2matrices([
            [[0, 0, 0, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 0]], [[0, 0, 1, 1]], [[0, 1, 0, 0]], [[0, 1, 0, 1]], [[0, 1, 1, 0]], [[0, 1, 1, 1]],
            [[1, 0, 0, 0]], [[1, 0, 0, 1]], [[1, 0, 1, 0]], [[1, 0, 1, 1]], [[1, 1, 0, 0]], [[1, 1, 0, 1]], [[1, 1, 1, 0]], [[1, 1, 1, 1]]
        ].iter().map(array2vec3d).collect())
    };

    let labels = Tensor {
        value: Tensor::vec2matrices([
            [[0, 0, 0, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 0]], [[0, 0, 1, 1]], [[0, 1, 0, 0]], [[0, 1, 0, 1]], [[0, 1, 1, 0]], [[0, 1, 1, 1]],
            [[1, 0, 0, 0]], [[1, 0, 0, 1]], [[1, 0, 1, 0]], [[1, 0, 1, 1]], [[1, 1, 0, 0]], [[1, 1, 0, 1]], [[1, 1, 1, 0]], [[1, 1, 1, 1]]
        ].iter().map(array2vec3d).collect())
    };

    let mut resnet = ResNet::new(&inputs, &labels, seed);
    resnet.train(lr, epochs);
    let value = inputs.value.iter().map(|x| ResNet::threshold(&resnet.forward(x))).collect();
    let outputs = Tensor { value };
    println!();
    println!("{}", outputs);

}
