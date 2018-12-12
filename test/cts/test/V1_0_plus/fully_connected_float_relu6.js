describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Fully connected float relu6 example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-3, 0, 16];
    let op3_expect = [0, 4, 6];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type0);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([2]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([3]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
