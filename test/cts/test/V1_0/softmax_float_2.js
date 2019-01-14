describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax float example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0];
    let output_expect = [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647, 0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1.0]));
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
