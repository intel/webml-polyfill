// Generated file (from: avg_pool_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Avg pool v1_2 example-1', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 2.0, 3.0, 4.0];
    let op4_expect = [1.0, 2.0, 3.0, 4.0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let param = operandIndex++;
    model.addOperand(type2);
    let param1 = operandIndex++;
    model.addOperand(type2);
    let param2 = operandIndex++;
    model.addOperand(type2);
    let param3 = operandIndex++;
    model.addOperand(type2);
    let param4 = operandIndex++;
    model.addOperand(type2);
    let param5 = operandIndex++;
    model.addOperand(type2);
    let param6 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type2);
    let param8 = operandIndex++;
    model.addOperand(type2);
    let op4 = operandIndex++;
    model.addOperand(type1);

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, param, param1, param2, param3, param4, param5, param6, param7, param8], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-2', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 2.0, 3.0, 4.0];
    let op4_expect = [1.0, 2.0, 3.0, 4.0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let param = operandIndex++;
    model.addOperand(type2);
    let param1 = operandIndex++;
    model.addOperand(type2);
    let param2 = operandIndex++;
    model.addOperand(type2);
    let param3 = operandIndex++;
    model.addOperand(type2);
    let param4 = operandIndex++;
    model.addOperand(type2);
    let param5 = operandIndex++;
    model.addOperand(type2);
    let param6 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type2);
    let param8 = operandIndex++;
    model.addOperand(type2);
    let op4 = operandIndex++;
    model.addOperand(type1);

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, param, param1, param2, param3, param4, param5, param6, param7, param8], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-3', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 2.0, 3.0, 4.0];
    let op4_expect = [1.0, 2.0, 3.0, 4.0];

    let type19 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type19);
    let param = operandIndex++;
    model.addOperand(type2);
    let param1 = operandIndex++;
    model.addOperand(type2);
    let param2 = operandIndex++;
    model.addOperand(type2);
    let param3 = operandIndex++;
    model.addOperand(type2);
    let param4 = operandIndex++;
    model.addOperand(type2);
    let param5 = operandIndex++;
    model.addOperand(type2);
    let param6 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type2);
    let param8 = operandIndex++;
    model.addOperand(type2);
    let op4 = operandIndex++;
    model.addOperand(type19);

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, param, param1, param2, param3, param4, param5, param6, param7, param8], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type19_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type19_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-4', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 4, 6, 8];
    let op4_expect = [2, 4, 6, 8];

    let type2 = {type: nn.INT32};
    let type20 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.5, zeroPoint: 0};
    let type20_length = product(type20.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type20);
    let param = operandIndex++;
    model.addOperand(type2);
    let param1 = operandIndex++;
    model.addOperand(type2);
    let param2 = operandIndex++;
    model.addOperand(type2);
    let param3 = operandIndex++;
    model.addOperand(type2);
    let param4 = operandIndex++;
    model.addOperand(type2);
    let param5 = operandIndex++;
    model.addOperand(type2);
    let param6 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type2);
    let param8 = operandIndex++;
    model.addOperand(type2);
    let op4 = operandIndex++;
    model.addOperand(type20);

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, param, param1, param2, param3, param4, param5, param6, param7, param8], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type20_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type20_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-5', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [0, 6, 2, 4, 3, 2, 10, 7];
    let op44_expect = [2.75, 5.75];

    let type2 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 4, 1]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 1]};
    let type8_length = product(type8.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type7);
    let param36 = operandIndex++;
    model.addOperand(type2);
    let param37 = operandIndex++;
    model.addOperand(type2);
    let param38 = operandIndex++;
    model.addOperand(type2);
    let param39 = operandIndex++;
    model.addOperand(type2);
    let param40 = operandIndex++;
    model.addOperand(type2);
    let param41 = operandIndex++;
    model.addOperand(type2);
    let op44 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(param36, new Int32Array([1]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([2]));
    model.setOperandValue(param39, new Int32Array([2]));
    model.setOperandValue(param40, new Int32Array([2]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op14, param36, param37, param38, param39, param40, param41], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type8_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-6', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_relaxed_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [0, 6, 2, 4, 3, 2, 10, 7];
    let op44_expect = [2.75, 5.75];

    let type2 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 4, 1]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 1]};
    let type8_length = product(type8.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type7);
    let param36 = operandIndex++;
    model.addOperand(type2);
    let param37 = operandIndex++;
    model.addOperand(type2);
    let param38 = operandIndex++;
    model.addOperand(type2);
    let param39 = operandIndex++;
    model.addOperand(type2);
    let param40 = operandIndex++;
    model.addOperand(type2);
    let param41 = operandIndex++;
    model.addOperand(type2);
    let op44 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(param36, new Int32Array([1]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([2]));
    model.setOperandValue(param39, new Int32Array([2]));
    model.setOperandValue(param40, new Int32Array([2]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op14, param36, param37, param38, param39, param40, param41], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type8_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-7', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_float16_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [0.0, 6.0, 2.0, 4.0, 3.0, 2.0, 10.0, 7.0];
    let op44_expect = [2.75, 5.75];

    let type2 = {type: nn.INT32};
    let type44 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 4, 1]};
    let type44_length = product(type44.dimensions);
    let type45 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 1]};
    let type45_length = product(type45.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type44);
    let param36 = operandIndex++;
    model.addOperand(type2);
    let param37 = operandIndex++;
    model.addOperand(type2);
    let param38 = operandIndex++;
    model.addOperand(type2);
    let param39 = operandIndex++;
    model.addOperand(type2);
    let param40 = operandIndex++;
    model.addOperand(type2);
    let param41 = operandIndex++;
    model.addOperand(type2);
    let op44 = operandIndex++;
    model.addOperand(type45);

    model.setOperandValue(param36, new Int32Array([1]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([2]));
    model.setOperandValue(param39, new Int32Array([2]));
    model.setOperandValue(param40, new Int32Array([2]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op14, param36, param37, param38, param39, param40, param41], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type45_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type45_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Avg pool v1_2 example-8', async function() {
    // For 'Avg pool v1_2' example: examples_nhwc_quant8_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [0, 24, 8, 16, 12, 8, 40, 28];
    let op44_expect = [11, 23];

    let type2 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 4, 1], scale: 0.25, zeroPoint: 0};
    let type46_length = product(type46.dimensions);
    let type47 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 2, 1], scale: 0.25, zeroPoint: 0};
    let type47_length = product(type47.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type46);
    let param36 = operandIndex++;
    model.addOperand(type2);
    let param37 = operandIndex++;
    model.addOperand(type2);
    let param38 = operandIndex++;
    model.addOperand(type2);
    let param39 = operandIndex++;
    model.addOperand(type2);
    let param40 = operandIndex++;
    model.addOperand(type2);
    let param41 = operandIndex++;
    model.addOperand(type2);
    let op44 = operandIndex++;
    model.addOperand(type47);

    model.setOperandValue(param36, new Int32Array([1]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([2]));
    model.setOperandValue(param39, new Int32Array([2]));
    model.setOperandValue(param40, new Int32Array([2]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op14, param36, param37, param38, param39, param40, param41], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Uint8Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Uint8Array(type47_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type47_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op44_output[i], op44_expect[i]));
    }
  });
});
