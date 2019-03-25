describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Batch to space example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1];
    let output_expect = [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 1, 2]};
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
