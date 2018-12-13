describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax float example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1.0, 2.0, 10.0, 20.0];
    let output_expect = [0.25, 0.25, 0.25, 0.25];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1e-06]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type0_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
