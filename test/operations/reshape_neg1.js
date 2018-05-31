describe('Reshape Test using special value -1', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it.skip('check result', async function() {
    let model = await nn.createModel(options);

    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions:[1, 4]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
    model.setOperandValue(1, new Int32Array([2, -1]));
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, -1]});
    model.addOperation(nn.RESHAPE, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData = new Float32Array(tensorLength);
    inputData.set([1.0, 2.0, 3.0, 4.0]);
    execution.setInput(0, inputData);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], inputData[i]));
    }
  });
});
