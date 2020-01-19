// Generated file (from: transpose_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Transpose v1_2 example-1', async function() {
    // For 'Transpose v1_2' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1.0, 2.0, 3.0, 4.0];
    let perms_value = [1, 0];
    let output_expect = [1.0, 3.0, 2.0, 4.0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let perms = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(perms, new Int32Array(perms_value));

    model.addOperation(nn.TRANSPOSE, [input, perms], [output]);

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

  it('check result for Transpose v1_2 example-2', async function() {
    // For 'Transpose v1_2' example: examples_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [1.0, 2.0, 3.0, 4.0];
    let perms_value = [1, 0];
    let output_expect = [1.0, 3.0, 2.0, 4.0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let perms = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(perms, new Int32Array(perms_value));

    model.addOperation(nn.TRANSPOSE, [input, perms], [output]);

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

  it('check result for Transpose v1_2 example-3', async function() {
    // For 'Transpose v1_2' example: examples_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [2, 4, 6, 8];
    let perms_value = [1, 0];
    let output_expect = [2, 6, 4, 8];

    let type1 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type1_length = product(type1.dimensions);
    let type14 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2], scale: 0.5, zeroPoint: 0};
    let type14_length = product(type14.dimensions);

    let input = operandIndex++;
    model.addOperand(type14);
    let perms = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type14);

    model.setOperandValue(perms, new Int32Array(perms_value));

    model.addOperation(nn.TRANSPOSE, [input, perms], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Uint8Array(input_value);
    execution.setInput(0, input_input);
    let output_output = new Uint8Array(type14_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type14_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
