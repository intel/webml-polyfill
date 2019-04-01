describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Fully connected quant8 example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [129, 131, 133, 135, 137, 139, 141, 143, 109, 107, 129, 131, 133, 135, 137, 139, 141, 111, 145, 107];
    let op3_expect = [151, 152, 153, 185, 186, 187];

    let type4 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.25, zeroPoint: 0};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 3], scale: 1., zeroPoint: 127};
    let type3_length = product(type3.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 10], scale: 0.5, zeroPoint: 127};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 1, 5, 1], scale: 0.5, zeroPoint: 127};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let act_relu = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Uint8Array([129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147]));
    model.setOperandValue(b0, new Int32Array([4, 8, 12]));
    model.setOperandValue(act_relu, new Int32Array([1]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act_relu], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Uint8Array(type3_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
