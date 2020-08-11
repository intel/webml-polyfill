
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

    // constant1 is a constant tensor. Set its value from an ArrayBuffer object.
    // The ArrayBuffer object may contain the training data loaded before hand.
    const constant1Value = new Float32Array(arrayBuffer, 0, this.tensorSize_);
    const constant1 = nn.constant(float32TensorType,
                                  constant1Value);

    // input1 is one of the input tensors. Its value will be set before execution.
    const input1 = nn.input('input1', float32TensorType);

    // constant2 is another constant tensor. Set its value from same ArrayBuffer
    // object with offset.
    const constant2Value = new Float32Array(arrayBuffer,
                                            this.tensorSize_ * Float32Array.BYTES_PER_ELEMENT,
                                            this.tensorSize_)
    const constant2 = nn.constant(float32TensorType,
                                  constant2Value);

    // input2 is another input tensor. Its value will be set before execution.
    const input2 = nn.input('input2', float32TensorType);

    // intermediateOutput0 is the output of the first Add operation.
    const intermediateOutput0 = nn.add(constant1, input1);

    // intermediateOutput1 is the output of the second Add operation.
    const intermediateOutput1 = nn.add(constant2, input2);

    // output is the output tensor of the Mul operation.
    const output = nn.mul(intermediateOutput0, intermediateOutput1);

    this.model_ = await nn.createModel([{name: 'output', operand: output}]);

    return [constant1Value[0], constant2Value[0]];
  }

  async compile(options) {
    this.compilation_ = await this.model_.createCompilation(options);
  }

  async compute(inputValue1, inputValue2) {
    // Create an Execution object for the compiled model.
    const execution = await this.compilation_.createExecution();

    const inputBuffer1 = new Float32Array(this.tensorSize_);
    inputBuffer1.fill(inputValue1);
    const inputBuffer2 = new Float32Array(this.tensorSize_);
    inputBuffer2.fill(inputValue2);

    // Associate the input buffers to model's inputs.
    execution.setInput('input1', inputBuffer1);
    execution.setInput('input2', inputBuffer2);

    // Associate the output buffer to model's output.
    const outputTensor = new Float32Array(this.tensorSize_);
    execution.setOutput('output', outputTensor);

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
