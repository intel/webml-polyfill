// Generated file (from: resize_bilinear_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Resize bilinear v1_2 example-1', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0];
    let op4_expect = [1.0, 1.0, 1.0, 1.666666667, 1.666666667, 1.666666667, 2.0, 2.0, 2.0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let param = operandIndex++;
    model.addOperand(type3);
    let param1 = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(param, new Int32Array([3]));
    model.setOperandValue(param1, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, param, param1], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type2_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-2', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0];
    let op4_expect = [1.0, 1.0, 1.0, 1.666666667, 1.666666667, 1.666666667, 2.0, 2.0, 2.0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let param = operandIndex++;
    model.addOperand(type3);
    let param1 = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(param, new Int32Array([3]));
    model.setOperandValue(param1, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, param, param1], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type2_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-3', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0];
    let op4_expect = [1.0, 1.0, 1.0, 1.6666666269302368, 1.6666666269302368, 1.6666666269302368, 2.0, 2.0, 2.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type17_length = product(type17.dimensions);
    let type3 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type16);
    let param = operandIndex++;
    model.addOperand(type3);
    let param1 = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param, new Int32Array([3]));
    model.setOperandValue(param1, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, param, param1], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type17_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-4', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [3, 4, 6, 10, 9, 10, 12, 16];
    let op41_expect = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];

    let type3 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let param4 = operandIndex++;
    model.addOperand(type3);
    let param5 = operandIndex++;
    model.addOperand(type3);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(param4, new Int32Array([3]));
    model.setOperandValue(param5, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op11, param4, param5], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type6_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-5', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [3, 4, 6, 10, 9, 10, 12, 16];
    let op41_expect = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];

    let type3 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let param4 = operandIndex++;
    model.addOperand(type3);
    let param5 = operandIndex++;
    model.addOperand(type3);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(param4, new Int32Array([3]));
    model.setOperandValue(param5, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op11, param4, param5], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type6_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-6', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_nhwc_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [3.0, 4.0, 6.0, 10.0, 9.0, 10.0, 12.0, 16.0];
    let op41_expect = [3.0, 4.0, 5.0, 8.0, 6.0, 10.0, 7.0, 8.0, 9.0, 12.0, 10.0, 14.0, 9.0, 10.0, 11.0, 14.0, 12.0, 16.0];

    let type27 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type27_length = product(type27.dimensions);
    let type28 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type28_length = product(type28.dimensions);
    let type3 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type27);
    let param4 = operandIndex++;
    model.addOperand(type3);
    let param5 = operandIndex++;
    model.addOperand(type3);
    let op41 = operandIndex++;
    model.addOperand(type28);

    model.setOperandValue(param4, new Int32Array([3]));
    model.setOperandValue(param5, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op11, param4, param5], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type28_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type28_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Resize bilinear v1_2 example-7', async function() {
    // For 'Resize bilinear v1_2' example: examples_shape_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [1.0, 1.0, 2.0, 2.0];
    let op42_expect = [1.0, 1.0, 1.0, 1.6666666269302368, 1.6666666269302368, 1.6666666269302368, 2.0, 2.0, 2.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type17_length = product(type17.dimensions);
    let type3 = {type: nn.INT32};

    let op12 = operandIndex++;
    model.addOperand(type16);
    let param8 = operandIndex++;
    model.addOperand(type3);
    let param9 = operandIndex++;
    model.addOperand(type3);
    let op42 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(param8, new Int32Array([3]));
    model.setOperandValue(param9, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op12, param8, param9], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type17_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });
});
