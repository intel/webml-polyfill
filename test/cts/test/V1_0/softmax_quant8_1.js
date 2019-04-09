describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Softmax quant8 example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1, 2, 10, 20];
    let output_expect = [64, 64, 64, 64];

    let type1 = {type: nn.FLOAT32};
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 4], scale: 0.00390625, zeroPoint: 0};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 4], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(beta, new Float32Array([1e-05]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Uint8Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Uint8Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
