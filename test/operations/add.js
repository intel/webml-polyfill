describe('Add Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  const nn = navigator.ml.nn;
  const value0 = 0.4;
  const value1 = 0.5;
    
  it('check result', async function() {
    let model = new nn.Model('SimpleModel');
    const float32TensorType = {type: 'tensor_float32', dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = model.addOperand({type: 'int32'});
    model.setOperandValue(fusedActivationFuncNone, nn.FuseCode.none);

    let input0 = model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = model.addOperand(float32TensorType);
    let output = model.addOperand(float32TensorType);

    model.addOperation('add', [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    model.finish();

    let compilation = new nn.Compilation(model);

    compilation.setPreference('fast_single_answer');
    
    await compilation.finish();

    let execution = new nn.Execution(compilation);

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], input0Data[i] + input1Data[i]));
    }
  });
});