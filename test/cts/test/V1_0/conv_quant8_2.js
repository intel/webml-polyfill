describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv quant8 example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [133, 131, 129, 125, 123, 121, 135, 133, 131, 123, 121, 119, 137, 135, 133, 121, 119, 117];
    let op4_expect = [157, 103, 167, 93];

    let type3 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.25, zeroPoint: 0};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.5, zeroPoint: 127};
    let type1_length = product(type1.dimensions);
    let type4 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 1., zeroPoint: 127};
    let type4_length = product(type4.dimensions);
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 6, 1], scale: 0.5, zeroPoint: 127};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad_valid = operandIndex++;
    model.addOperand(type3);
    let act_none = operandIndex++;
    model.addOperand(type3);
    let stride1 = operandIndex++;
    model.addOperand(type3);
    let stride3 = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Uint8Array([129, 131, 133, 135]));
    model.setOperandValue(op3, new Int32Array([-4]));
    model.setOperandValue(pad_valid, new Int32Array([2]));
    model.setOperandValue(act_none, new Int32Array([0]));
    model.setOperandValue(stride1, new Int32Array([1]));
    model.setOperandValue(stride3, new Int32Array([3]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad_valid, stride3, stride1, act_none], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Uint8Array(type4_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
