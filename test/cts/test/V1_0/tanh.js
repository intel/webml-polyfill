describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Tanh example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-1, 0, 1, 10];
    let op2_expect = [-0.761594156, 0, 0.761594156, 0.999999996];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type0);

    model.addOperation(nn.TANH, [op1], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type0_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
