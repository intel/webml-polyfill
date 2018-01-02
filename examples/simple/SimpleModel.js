const nn = navigator.ml.nn;

const TENSOR_SIZE = 200;
const FLOAT_EPISILON = 1e-6;

class SimpleModel {
  constructor(arrayBuffer) {
    this.arrayBuffer_ = arrayBuffer;
    this.tensorSize_ = TENSOR_SIZE;
    this.model_ = null;
  }

  createCompiledModel() {
    // create a Model.
    this.model_ = new nn.Model('SimpleModel');

    let float32TensorType = {type: 'tensor-float32', dimensions: [TENSOR_SIZE]};
    let scalarInt32Type = {type: 'int32'};

    // We first add the operand for the NONE activation function, and set its
    // value to FUSED_NONE.
    // This constant scalar operand will be used for all 3 operations.
    let fusedActivationFuncNone = this.model_.addOperand(scalarInt32Type);
    this.model_.setOperandValue(fusedActivationFuncNone, new Float32Array([nn.FuseCode.FUSED_NONE]));

    // tensor0 is a constant tensor that was established during training.
    // We read these values from the corresponding memory object.
    let tensor0 = this.model_.addOperand(float32TensorType);
    this.model_.setOperandValue(tensor0, new Float32Array(this.arrayBuffer_, 0, TENSOR_SIZE));

    // tensor1 is one of the user provided input tensors to the trained this.model_.
    // Its value is determined pre-execution.
    let tensor1 = this.model_.addOperand(float32TensorType);

    // tensor2 is a constant tensor that was established during training.
    // We read these values from the corresponding memory object.
    let tensor2 = this.model_.addOperand(float32TensorType);
    this.model_.setOperandValue(tensor2, new Float32Array(this.arrayBuffer_, TENSOR_SIZE * Float32Array.BYTES_PER_ELEMENT, TENSOR_SIZE));

    // tensor3 is one of the user provided input tensors to the trained this.model_.
    // Its value is determined pre-execution.
    let tensor3 = this.model_.addOperand(float32TensorType);

    // intermediateOutput0 is the output of the first ADD operation.
    // Its value is computed during execution.
    let intermediateOutput0 = this.model_.addOperand(float32TensorType);

    // intermediateOutput1 is the output of the second ADD operation.
    // Its value is computed during execution.
    let intermediateOutput1 = this.model_.addOperand(float32TensorType);

    // multiplierOutput is the output of the MUL operation.
    // Its value will be computed during execution.
    let multiplierOutput = this.model_.addOperand(float32TensorType);

    // Add the first ADD operation.
    this.model_.addOperation('add', [tensor0, tensor1, fusedActivationFuncNone], [intermediateOutput0]);

    // Add the second ADD operation.
    // Note the fusedActivationFuncNone is used again.
    this.model_.addOperation('add', [tensor2, tensor3, fusedActivationFuncNone], [intermediateOutput1]);

    // Add the MUL operation.
    // Note that intermediateOutput0 and intermediateOutput1 are specified
    // as inputs to the operation.
    this.model_.addOperation('mul', [intermediateOutput0, intermediateOutput1, fusedActivationFuncNone], [multiplierOutput]);

    // Identify the input and output tensors to the this.model_.
    // Inputs: {tensor1, tensor3}
    // Outputs: {multiplierOutput}
    this.model_.identifyInputsAndOutputs([tensor1, tensor3], [multiplierOutput]);

    // Finish constructing the this.model_.
    // The values of constant and intermediate operands cannot be altered after
    // the finish function is called.
    this.model_.finish();

    // Create a Compilation object for the constructed this.model_.
    this.compilation_ = new nn.Compilation(this.model_);

    // Set the preference for the compilation, so that the runtime and drivers
    // can make better decisions.
    // Here we prefer to get the answer quickly, so we choose
    // FAST_SINGLE_ANSWER.
    this.compilation_.setPreference('fast-single-answer');

    // Finish the compilation.
    this.compilation_.finish();
  }

  async compute(inputValue1, inputValue2) {
    let execution = new nn.Execution(this.compilation_);
    let inputTensor1 = new Float32Array(this.tensorSize_);
    inputTensor1.fill(inputValue1);
    let inputTensor2 = new Float32Array(this.tensorSize_);
    inputTensor2.fill(inputValue2);

    // Tell the execution to associate inputTensor1 to the first of the two model inputs.
    // Note that the index of the modelInput list {tensor1, tensor3}
    execution.setInput(0, inputTensor1);
    execution.setInput(1, inputTensor2);

    let outputTensor = new Float32Array(this.tensorSize_);
    execution.setOutput(0, outputTensor);

    await execution.startCompute();

    const goldenRef = (inputValue1 + 0.5) * (inputValue2 + 0.5);
    for (let i = 0; i < outputTensor.length; ++i) {
      let delta = Math.abs(outputTensor[i] - goldenRef);
      if (delta > FLOAT_EPISILON) {
        console.error(`Output computation error: output(${outputTensor[i]}), delta(${delta}) @ idx(${i})`)
      }
    }

    return outputTensor[0];
  }
}
