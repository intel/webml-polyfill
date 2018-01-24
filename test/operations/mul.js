describe('Mul Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  const nn = navigator.ml.getNeuralNetworkContext();
  const value0 = 0.4;
  const value1 = 0.5;
    
  it('check result', async function() {
    let model = nn.createModel();
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = model.addOperand(float32TensorType);
    let output = model.addOperand(float32TensorType);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    model.finish();

    let compilation = nn.createCompilation(model);

    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    
    await compilation.finish();

    let execution = nn.createExecution(compilation);

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], input0Data[i] * input1Data[i]));
    }
  });
});