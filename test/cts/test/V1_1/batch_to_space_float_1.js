describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Batch to space float example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let output_expect = [1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 4, 1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let block_size = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(block_size, new Int32Array([2, 2]));
    model.addOperation(nn.BATCH_TO_SPACE_ND, [input, block_size], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
