describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Mul broadcasting 4D-4D example/6', async function() {
    let model = await nn.createModel(options);
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4]);

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
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  20,  40,  60,  80,
                         30,  60,  90, 120,  40,  80, 120, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
});
