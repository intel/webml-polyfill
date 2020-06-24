
var nn = navigator.ml.getNeuralNetworkContext('v2');

class SimpleModel {
  constructor(url) {
    this.url_ = url;
    this.tensorSize_ = 200;
    this.model_ = null;
    this.compilation_ = null;
  }

  async load() {
    const response = await fetch(this.url_);
    const arrayBuffer = await response.arrayBuffer();

    // Create OperandDescriptor object.
    const float32TensorType = {type: 'tensor-float32', dimensions: [this.tensorSize_]};

    // tensor0 is a constant tensor. Set its value from an ArrayBuffer object.
    // The ArrayBuffer object may contain the training data loaded before hand.
    const tensor0 = nn.constant(float32TensorType,
                                new Float32Array(arrayBuffer, 0, this.tensorSize_));

    // tensor1 is one of the input tensors. Its value will be set before execution.
    const tensor1 = nn.input(float32TensorType);

    // tensor2 is another constant tensor. Set its value from same ArrayBuffer
    // object with offset.
    const tensor2 = nn.constant(float32TensorType,
                                new Float32Array(arrayBuffer,
                                                 this.tensorSize_ * Float32Array.BYTES_PER_ELEMENT,
                                                 this.tensorSize_));

    // tensor3 is another input tensor. Its value will be set before execution.
    const tensor3 = nn.input(float32TensorType);

    // intermediateOutput0 is the output of the first Add operation.
    const intermediateOutput0 = nn.add(tensor0, tensor1);

    // intermediateOutput1 is the output of the second Add operation.
    const intermediateOutput1 = nn.add(tensor2, tensor3);

    // output is the output tensor of the Mul operation.
    const output = nn.mul(intermediateOutput0, intermediateOutput1);

    this.model_ = await nn.createModel([output])
  }

  async compile(options) {
    this.compilation_ = await this.model_.createCompilation(options);
  }

  async compute(inputValue1, inputValue2) {
    const execution = await this.compilation_.createExecution();
    const inputTensor1 = new Float32Array(this.tensorSize_);
    inputTensor1.fill(inputValue1);
    const inputTensor2 = new Float32Array(this.tensorSize_);
    inputTensor2.fill(inputValue2);

    // Tell the execution to associate inputTensor1 to the first of the two model inputs.
    // Note that the index of the modelInput list {tensor1, tensor3}
    execution.setInput(0, inputTensor1);
    execution.setInput(1, inputTensor2);

    const outputTensor = new Float32Array(this.tensorSize_);
    execution.setOutput(0, outputTensor);

    await execution.startCompute();

    this.validate(inputValue1, inputValue2, outputTensor);
    return outputTensor[0];
  }

  validate(inputValue1, inputValue2, outputTensor) {
    const FLOAT_EPISILON = 1e-6;
    const goldenRef = (inputValue1 + 0.5) * (inputValue2 + 0.5);
    for (let i = 0; i < outputTensor.length; ++i) {
      let delta = Math.abs(outputTensor[i] - goldenRef);
      if (delta > FLOAT_EPISILON) {
        console.error(`Output computation error: output(${outputTensor[i]}), delta(${delta}) @ idx(${i})`)
      }
    }
  }
}
