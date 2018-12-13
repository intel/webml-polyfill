describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Mul broadcasting 3D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);

    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
                         7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12,
                        13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
                        19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
                        25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30,
                        31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36,
                        37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42,
                        43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48,
                        49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54,
                        55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60];


    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
});
