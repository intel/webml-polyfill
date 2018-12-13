describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('check result for Concatenation axis 2 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 2;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0,
                      201.0, 202.0, 203.0, 204.0,
                      105.0, 106.0, 107.0, 108.0,
                      205.0, 206.0, 207.0, 208.0,
                      109.0, 110.0, 111.0, 112.0,
                      209.0, 210.0, 211.0, 212.0,
                      113.0, 114.0, 115.0, 116.0,
                      213.0, 214.0, 215.0, 216.0]);
    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
});
