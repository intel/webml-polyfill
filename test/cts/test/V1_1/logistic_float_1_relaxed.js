// Generated file (from: logistic_float_1_relaxed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Logistic float 1 relaxed example', async function() {
    // For 'Logistic float 1 relaxed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 2.0, 4.0, 8.0];
    let op3_expect = [0.7310585975646973, 0.8807970285415649, 0.9820137619972229, 0.9996646642684937];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.addOperation(nn.LOGISTIC, [op1], [op3]);

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
