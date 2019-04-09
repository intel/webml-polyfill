describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Avg pool quant8 example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [0, 96, 32, 64, 48, 32, 160, 112];
    let op3_expect = [44, 92];

    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 2, 1], scale: 0.0625, zeroPoint: 0};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 4, 1], scale: 0.0625, zeroPoint: 0};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let cons2 = operandIndex++;
    model.addOperand(type1);
    let pad_same = operandIndex++;
    model.addOperand(type1);
    let act_none = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(cons2, new Int32Array([2]));
    model.setOperandValue(pad_same, new Int32Array([1]));
    model.setOperandValue(act_none, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, pad_same, cons2, cons2, cons2, cons2, act_none], [op3]);

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
