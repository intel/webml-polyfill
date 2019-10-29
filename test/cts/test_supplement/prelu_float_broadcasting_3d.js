describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for PRELU float broadcasting 3D example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.1, 2.2, 3.3, 4.4, -5.5, -6.6, -7.7, -8.8];
    let op2_value = [2];
    let op3_expect = [1.1, 2.2, 3.3, 4.4, -11.0, -13.2, -15.4, -17.6];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type0);

    let op2_input = new Float32Array(op2_value);
    model.setOperandValue(op2, op2_input);

    model.addOperation(nn.PRELU, [op1, op2], [op3]);

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
