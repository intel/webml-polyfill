describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Mul quant8 example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 2];
    let op2_value = [2, 4];
    let op3_expect = [1, 4];

    let type1 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2], scale: 1.0, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2], scale: 2.0, zeroPoint: 0};
    let type2_length = product(type2.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type0);
    let act = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);

    let op2_input = new Uint8Array(op2_value);
    model.setOperandValue(op2, op2_input);

    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.MUL, [op1, op2, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Uint8Array(type2_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
