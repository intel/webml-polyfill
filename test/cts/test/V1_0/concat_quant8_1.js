describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Concat quant8 example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 2, 3, 4, 5, 6];
    let op2_value = [7, 8, 9, 10, 11, 12];
    let result_expect = [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12];

    let type1 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 3], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 6], scale: 0.5, zeroPoint: 0};
    let type2_length = product(type2.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type0);
    let axis1 = operandIndex++;
    model.addOperand(type1);
    let result = operandIndex++;
    model.addOperand(type2);

    let op2_input = new Uint8Array(op2_value);
    model.setOperandValue(op2, op2_input);

    model.setOperandValue(axis1, new Int32Array([1]));
    model.addOperation(nn.CONCATENATION, [op1, op2, axis1], [result]);

    model.identifyInputsAndOutputs([op1], [result]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);

    let result_output = new Uint8Array(type2_length);
    execution.setOutput(0, result_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(result_output[i], result_expect[i]));
    }
  });
});
