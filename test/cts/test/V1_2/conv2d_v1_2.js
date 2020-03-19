// Generated file (from: conv2d_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv2d v1_2 example-1', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    model.setOperandValue(op3, new Float32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

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

  it('check result for Conv2d v1_2 example-2', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    model.setOperandValue(op3, new Float32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

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

  it('check result for Conv2d v1_2 example-3', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type33);
    let op3 = operandIndex++;
    model.addOperand(type34);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Uint8Array([2, 2, 2, 2]));
    model.setOperandValue(op3, new Int32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-4', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type36_length = product(type36.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.125])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array([2, 2, 2, 2]));
    model.setOperandValue(op3, new Int32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-5', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type37 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type37_length = product(type37.dimensions);
    let type38 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type37);
    let op2 = operandIndex++;
    model.addOperand(type38);
    let op3 = operandIndex++;
    model.addOperand(type39);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type38);

    model.setOperandValue(op2, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    model.setOperandValue(op3, new Float32Array([0.0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type38_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type38_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-6', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op2_value = [0.25, 0.25, 0.25, 0.25];
    let op3_value = [0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array(op2_value));
    model.setOperandValue(op3, new Float32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

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

  it('check result for Conv2d v1_2 example-7', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op2_value = [0.25, 0.25, 0.25, 0.25];
    let op3_value = [0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type1);
    let op2 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array(op2_value));
    model.setOperandValue(op3, new Float32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

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

  it('check result for Conv2d v1_2 example-8', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op2_value = [2, 2, 2, 2];
    let op3_value = [0];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type33);
    let op3 = operandIndex++;
    model.addOperand(type34);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Uint8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-9', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op2_value = [2, 2, 2, 2];
    let op3_value = [0];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type35 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type35_length = product(type35.dimensions);
    let type36 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type36_length = product(type36.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type35);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.125])});
    let op3 = operandIndex++;
    model.addOperand(type36);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-10', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let op2_value = [0.25, 0.25, 0.25, 0.25];
    let op3_value = [0.0];
    let op4_expect = [0.875, 0.875, 0.875, 0.875];

    let type37 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type37_length = product(type37.dimensions);
    let type38 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type37);
    let op2 = operandIndex++;
    model.addOperand(type38);
    let op3 = operandIndex++;
    model.addOperand(type39);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type38);

    model.setOperandValue(op2, new Float32Array(op2_value));
    model.setOperandValue(op3, new Float32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type38_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type38_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-11', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op41_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type5_length = product(type5.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type5);

    model.setOperandValue(op21, new Float32Array([1, 4, 7, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op31, new Float32Array([-200]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type5_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-12', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op41_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type5_length = product(type5.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type5);

    model.setOperandValue(op21, new Float32Array([1, 4, 7, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op31, new Float32Array([-200]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type5_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-13', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type47 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 127};
    let type47_length = product(type47.dimensions);
    let type48 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type47);
    let op31 = operandIndex++;
    model.addOperand(type48);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Uint8Array([129, 135, 141, 131, 137, 143, 133, 139, 145]));
    model.setOperandValue(op31, new Int32Array([-800]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-14', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type50_length = product(type50.dimensions);
    let type51 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type51_length = product(type51.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type50);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5])});
    let op31 = operandIndex++;
    model.addOperand(type51);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Int8Array([2, 8, 14, 4, 10, 16, 6, 12, 18]));
    model.setOperandValue(op31, new Int32Array([-800]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-15', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let op41_expect = [0.0, 0.0, 0.0, 0.0, 35.0, 112.0, 157.0, 0.0, 0.0, 34.0, 61.0, 0.0];

    let type37 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type37_length = product(type37.dimensions);
    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type52 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type52_length = product(type52.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type52);
    let op21 = operandIndex++;
    model.addOperand(type37);
    let op31 = operandIndex++;
    model.addOperand(type39);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type52);

    model.setOperandValue(op21, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    model.setOperandValue(op31, new Float32Array([-200.0]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type52_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type52_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-16', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op21_value = [1, 4, 7, 2, 5, 8, 3, 6, 9];
    let op31_value = [-200];
    let op41_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type5_length = product(type5.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type5);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type5_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-17', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op21_value = [1, 4, 7, 2, 5, 8, 3, 6, 9];
    let op31_value = [-200];
    let op41_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type5_length = product(type5.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type5);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type5_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type5_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-18', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op21_value = [129, 135, 141, 131, 137, 143, 133, 139, 145];
    let op31_value = [-800];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type47 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 127};
    let type47_length = product(type47.dimensions);
    let type48 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type47);
    let op31 = operandIndex++;
    model.addOperand(type48);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Uint8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-19', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op21_value = [2, 8, 14, 4, 10, 16, 6, 12, 18];
    let op31_value = [-800];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type50_length = product(type50.dimensions);
    let type51 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type51_length = product(type51.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type50);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5])});
    let op31 = operandIndex++;
    model.addOperand(type51);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-20', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let op21_value = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let op31_value = [-200.0];
    let op41_expect = [0.0, 0.0, 0.0, 0.0, 35.0, 112.0, 157.0, 0.0, 0.0, 34.0, 61.0, 0.0];

    let type37 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type37_length = product(type37.dimensions);
    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type52 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type52_length = product(type52.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type52);
    let op21 = operandIndex++;
    model.addOperand(type37);
    let op31 = operandIndex++;
    model.addOperand(type39);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type52);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type52_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type52_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-21', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type6);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op22, new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]));
    model.setOperandValue(op32, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type6_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-22', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type6);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op22, new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]));
    model.setOperandValue(op32, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type6_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-23', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type58 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type58_length = product(type58.dimensions);
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type59_length = product(type59.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type58);
    let op32 = operandIndex++;
    model.addOperand(type59);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    model.setOperandValue(op32, new Int32Array([0, 0, 0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-24', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type60 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type60_length = product(type60.dimensions);
    let type61 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type61_length = product(type61.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type60);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type61);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Int8Array([1, 2, 3, 5, 6, 8, 12, 13, 15]));
    model.setOperandValue(op32, new Int32Array([0, 0, 0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-25', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type62 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type62_length = product(type62.dimensions);
    let type63 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type62);
    let op22 = operandIndex++;
    model.addOperand(type63);
    let op32 = operandIndex++;
    model.addOperand(type64);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type62);

    model.setOperandValue(op22, new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]));
    model.setOperandValue(op32, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type62_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type62_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-26', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op22_value = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
    let op32_value = [0.0, 0.0, 0.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type6);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type6_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-27', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op22_value = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
    let op32_value = [0.0, 0.0, 0.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type6);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type6_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-28', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op22_value = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let op32_value = [0, 0, 0];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type58 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type58_length = product(type58.dimensions);
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type59_length = product(type59.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type58);
    let op32 = operandIndex++;
    model.addOperand(type59);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Uint8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-29', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op22_value = [1, 2, 3, 5, 6, 8, 12, 13, 15];
    let op32_value = [0, 0, 0];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type60 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type60_length = product(type60.dimensions);
    let type61 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type61_length = product(type61.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type60);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op32 = operandIndex++;
    model.addOperand(type61);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-30', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [5.0, 5.0, 5.0];
    let op22_value = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
    let op32_value = [0.0, 0.0, 0.0];
    let op42_expect = [15.0, 37.5, 60.0];

    let type4 = {type: nn.INT32};
    let type62 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type62_length = product(type62.dimensions);
    let type63 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type62);
    let op22 = operandIndex++;
    model.addOperand(type63);
    let op32 = operandIndex++;
    model.addOperand(type64);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type62);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type62_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type62_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-31', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type9_length = product(type9.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type9);
    let op23 = operandIndex++;
    model.addOperand(type7);
    let op33 = operandIndex++;
    model.addOperand(type8);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op23, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    model.setOperandValue(op33, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type9_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-32', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type9_length = product(type9.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type9);
    let op23 = operandIndex++;
    model.addOperand(type7);
    let op33 = operandIndex++;
    model.addOperand(type8);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op23, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    model.setOperandValue(op33, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type9_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-33', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type59_length = product(type59.dimensions);
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type69 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 128};
    let type69_length = product(type69.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type69);
    let op33 = operandIndex++;
    model.addOperand(type59);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Uint8Array([130, 136, 142, 132, 138, 144, 134, 140, 146]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-34', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);
    let type71 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type71_length = product(type71.dimensions);
    let type72 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type72_length = product(type72.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type71);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op33 = operandIndex++;
    model.addOperand(type72);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Int8Array([2, 8, 14, 2, 5, 8, 6, 12, 18]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-35', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145];
    let op43_expect = [157, 163, 169, 193, 208, 223, 229, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type4 = {type: nn.INT32};
    let type73 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: 127};
    let type73_length = product(type73.dimensions);
    let type74 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type74_length = product(type74.dimensions);
    let type75 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type75_length = product(type75.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type73);
    let op23 = operandIndex++;
    model.addOperand(type74);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op33 = operandIndex++;
    model.addOperand(type75);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type73);

    model.setOperandValue(op23, new Int8Array([2, 8, 14, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type73_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type73_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-36', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type63 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type76 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type76_length = product(type76.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type76);
    let op23 = operandIndex++;
    model.addOperand(type63);
    let op33 = operandIndex++;
    model.addOperand(type64);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type76);

    model.setOperandValue(op23, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    model.setOperandValue(op33, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type76_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type76_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-37', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op23_value = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let op33_value = [0.0, 0.0, 0.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type9_length = product(type9.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type9);
    let op23 = operandIndex++;
    model.addOperand(type7);
    let op33 = operandIndex++;
    model.addOperand(type8);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type9_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-38', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op23_value = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let op33_value = [0.0, 0.0, 0.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type9_length = product(type9.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type9);
    let op23 = operandIndex++;
    model.addOperand(type7);
    let op33 = operandIndex++;
    model.addOperand(type8);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type9_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-39', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op23_value = [130, 136, 142, 132, 138, 144, 134, 140, 146];
    let op33_value = [0, 0, 0];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type59_length = product(type59.dimensions);
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type69 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 128};
    let type69_length = product(type69.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type69);
    let op33 = operandIndex++;
    model.addOperand(type59);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Uint8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-40', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op23_value = [2, 8, 14, 2, 5, 8, 6, 12, 18];
    let op33_value = [0, 0, 0];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);
    let type71 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type71_length = product(type71.dimensions);
    let type72 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type72_length = product(type72.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type71);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op33 = operandIndex++;
    model.addOperand(type72);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-41', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145];
    let op23_value = [2, 8, 14, 2, 5, 8, 3, 6, 9];
    let op33_value = [0, 0, 0];
    let op43_expect = [157, 163, 169, 193, 208, 223, 229, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type4 = {type: nn.INT32};
    let type73 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: 127};
    let type73_length = product(type73.dimensions);
    let type74 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type74_length = product(type74.dimensions);
    let type75 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type75_length = product(type75.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type73);
    let op23 = operandIndex++;
    model.addOperand(type74);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op33 = operandIndex++;
    model.addOperand(type75);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type73);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type73_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type73_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-42', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
    let op23_value = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let op33_value = [0.0, 0.0, 0.0];
    let op43_expect = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];

    let type4 = {type: nn.INT32};
    let type63 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 3]};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type76 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 3, 3]};
    let type76_length = product(type76.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type76);
    let op23 = operandIndex++;
    model.addOperand(type63);
    let op33 = operandIndex++;
    model.addOperand(type64);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type76);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type76_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type76_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-43', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op44_expect = [1.85284, -0.0393656, -0.127353, 1.43115, -0.302294, -1.0402, 0.655023, -0.587614, 1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -0.346357, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.104506, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, 1.42026, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, -0.343435, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, -1.46717, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494, 1.23741, -0.527402, -0.39954, -0.0128623, 1.3644, 0.985755, -0.718118, -0.1008, 1.24327];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type12_length = product(type12.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type12);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type12_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type12_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-43', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op44_expect = [-0.000614278, -1.21221, 0.443861, 0.102117, -2.52714, 1.47489, 0.173474, -0.237577, 1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, -1.6271, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.133762, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 1.22372, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, -0.838905, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 0.891823, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468, 0.166228, 0.434554, -2.57529, -0.958662, -2.23978, 2.66776, 0.542601, 1.76107, -1.08134];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type12_length = product(type12.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type12);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type12_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type12_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-44', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op44_expect = [1.85284, -0.0393656, -0.127353, 1.43115, -0.302294, -1.0402, 0.655023, -0.587614, 1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -0.346357, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.104506, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, 1.42026, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, -0.343435, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, -1.46717, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494, 1.23741, -0.527402, -0.39954, -0.0128623, 1.3644, 0.985755, -0.718118, -0.1008, 1.24327];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type12_length = product(type12.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type12);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type12_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type12_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-44', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op44_expect = [-0.000614278, -1.21221, 0.443861, 0.102117, -2.52714, 1.47489, 0.173474, -0.237577, 1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, -1.6271, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.133762, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 1.22372, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, -0.838905, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 0.891823, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468, 0.166228, 0.434554, -2.57529, -0.958662, -2.23978, 2.66776, 0.542601, 1.76107, -1.08134];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type12_length = product(type12.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type12);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type12_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type12_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-45', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.8699309825897217, 0.6446279883384705, -0.9183930158615112, 0.153671994805336, 0.8685619831085205, -0.3581770062446594, -0.13493099808692932, -0.24756500124931335, 0.2217400074005127, -0.2591570019721985, -0.2842960059642792, -0.5380650162696838, 0.7655590176582336, 0.41986000537872314, -0.556240975856781, 0.6584939956665039, 0.214355006814003, -0.8501690030097961, -0.25289300084114075, -0.47893500328063965, 0.5305259823799133, -0.07006630301475525, -0.9887290000915527, -0.30306100845336914, 0.150845006108284, 0.8299149870872498, 0.47634899616241455, 0.4065369963645935, -0.3553430140018463, 0.7571449875831604, -0.356361985206604, 0.8004819750785828, -0.7138609886169434, 0.21048299968242645, -0.6343029737472534, 0.7182360291481018, -0.7520380020141602, 0.45754700899124146, -0.5507689714431763, -0.551177978515625, 0.4467659890651703, -0.22746199369430542, 0.21634800732135773, -0.8528059720993042, -0.3514859974384308, 0.5590599775314331, -0.6684929728507996, -0.3034929931163788, -0.3637630045413971, -0.16283699870109558, 0.07010120153427124, 0.7560970187187195, -0.14226900041103363, 0.3297240138053894, -0.6563169956207275, -0.9980859756469727, -0.6529489755630493, -0.4031600058078766, -0.8936820030212402, 0.43274399638175964, 0.6123620271682739, -0.8695880174636841, -0.7132700085639954, -0.39809200167655945, -0.04235589876770973, 0.4365760087966919, -0.9252719879150391, 0.1765490025281906, 0.8229039907455444, 0.0968329980969429, -0.29680201411247253, -0.4271950125694275, 0.0316540002822876, -0.25447899103164673, 0.2449049949645996, 0.09482540190219879, 0.6437690258026123, -0.9039099812507629, 0.3526650071144104, -0.9011790156364441, 0.2661589980125427, -0.96806800365448, -0.615401029586792, -0.38897499442100525, 0.9390519857406616, -0.11628899723291397, 0.10752300173044205, -0.058271098881959915, 0.4351719915866852, 0.33467501401901245, 0.4597109854221344, 0.7174360156059265, 0.49662700295448303, -0.6801750063896179, -0.4150660037994385, 0.33984801173210144, 0.5060039758682251, -0.3378080129623413, -0.10721799731254578, -0.1724960058927536, 0.8706380128860474, 0.9318720102310181, -0.9538840055465698, 0.9030420184135437, 0.7600780129432678, 0.20972700417041779, -0.28538399934768677, -0.45513999462127686, 0.11319400370121002, 0.07566110044717789, 0.09244350343942642, -0.47286298871040344, 0.9606090188026428, -0.16038499772548676, -0.8394449949264526, 0.45709699392318726, 0.1633480042219162, 0.3448669910430908, -0.13161900639533997, 0.6887149810791016, -0.5408269762992859, 0.5712590217590332, -0.9558699727058411, 0.506164014339447, -0.1558389961719513, 0.07896210253238678, 0.756771981716156, -0.6620690226554871, 0.24290800094604492, 0.4608210027217865, 0.17787200212478638, -0.2898389995098114, -0.6406030058860779, 0.702597975730896, -0.5064060091972351, -0.568261981010437, -0.07137160003185272, 0.4137920141220093, 0.15967300534248352, -0.3052079975605011, 0.13381600379943848, -0.16025400161743164, 0.787322998046875, -0.7532439827919006, 0.600721001625061, 0.2631860077381134, -0.16238699853420258, 0.4779619872570038, -0.7029510140419006, -0.7310360074043274, -0.9394810199737549, -0.5245190262794495, 0.9340720176696777, -0.5116369724273682, -0.5034989714622498, 0.10623600333929062, -0.3236840069293976, 0.5344439744949341, -0.8437449932098389, 0.36417099833488464, 0.03703580051660538, -0.16880099475383759, -0.4045589864253998, -0.8141779899597168, 0.9174500107765198, -0.3342759907245636, 0.6692500114440918, -0.8012009859085083, 0.15651099383831024, -0.4279490113258362, 0.3791530132293701, 0.8185970187187195, -0.6499019861221313, 0.4270870089530945, -0.586014986038208, -0.5597890019416809, -0.8339229822158813, 0.0892409011721611, -0.6212509870529175, 0.2138260006904602, 0.46550899744033813, 0.47040000557899475, 0.38026100397109985, 0.4130670130252838, 0.1808219999074936, 0.17286600172519684, 0.5961400270462036, 0.8255749940872192, 0.6629160046577454, -0.704380989074707, -0.29763099551200867, 0.6977779865264893];
    let op44_expect = [1.8528399467468262, -0.03936560079455376, -0.1273529976606369, 1.431149959564209, -0.302293986082077, -1.0401999950408936, 0.6550229787826538, -0.5876139998435974, 1.7200299501419067, 1.5581599473953247, 0.6675459742546082, 2.2366299629211426, 0.06615159660577774, 0.29025399684906006, 0.770222008228302, -0.34635698795318604, -1.581969976425171, -0.8505949974060059, -0.48422399163246155, 0.9499670267105103, -0.5772629976272583, -0.8719490170478821, 2.341320037841797, -0.1045060008764267, -0.1359650045633316, -0.985713005065918, 0.8151469826698303, 1.0311399698257446, -1.4191499948501587, -0.515533983707428, -0.37363898754119873, 1.420259952545166, -1.5060399770736694, 0.6731129884719849, 3.061389923095703, -0.38857799768447876, -1.7670700550079346, -0.3156670033931732, -1.0381499528884888, -0.34343498945236206, 0.4327870011329651, -1.4164299964904785, 1.1294399499893188, -0.17580600082874298, -0.8464149832725525, 1.4009499549865723, 0.7083200216293335, -1.467170000076294, 2.195620059967041, -2.6126599311828613, -0.7053830027580261, 1.261240005493164, 1.4654500484466553, -2.357609987258911, 2.0449399948120117, 1.2374099493026733, -0.5274019837379456, -0.3995400071144104, -0.01286229956895113, 1.364400029182434, 0.9857550263404846, -0.7181180119514465, -0.10080000013113022, 1.2432700395584106];

    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type80_length = product(type80.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type82 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type82_length = product(type82.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type81);
    let op24 = operandIndex++;
    model.addOperand(type80);
    let op34 = operandIndex++;
    model.addOperand(type39);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type82);

    model.setOperandValue(op24, new Float32Array([-0.9662129878997803, -0.4674740135669708, -0.8220300078392029, -0.5794550180435181, 0.027880899608135223, -0.7994599938392639, -0.6842589974403381, 0.5632380247116089, 0.37288999557495117, 0.738215982913971, 0.38604500889778137, -0.9177749752998352, 0.18432499468326569, -0.27056801319122314, 0.8223599791526794, 0.09736829996109009, -0.9413080215454102, -0.14470599591732025]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type82_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type82_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-45', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_SAME_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.29533499479293823, -0.0038760099560022354, -0.5522509813308716, 0.16608400642871857, -0.28481999039649963, -0.15214300155639648, -0.719884991645813, -0.8693860173225403, -0.7455980181694031, 0.823947012424469, 0.4731830060482025, -0.33133700489997864, 0.18763099610805511, 0.04265709966421127, -0.8268970251083374, -0.7550849914550781, -0.4724529981613159, -0.023365600034594536, 0.04834359884262085, 0.9334179759025574, -0.961974024772644, 0.012578300200402737, 0.21974200010299683, 0.34260401129722595, -0.1516599953174591, 0.09349049627780914, 0.7832210063934326, 0.12966400384902954, 0.8388440012931824, -0.2713879942893982, 0.9245190024375916, 0.3428429961204529, 0.2744179964065552, 0.3508169949054718, 0.8416380286216736, -0.5439929962158203, -0.002833950100466609, -0.12846699357032776, -0.6829429864883423, -0.31911700963974, 0.846340000629425, 0.2830030024051666, 0.32864999771118164, 0.029375499114394188, -0.03356960043311119, 0.5912659764289856, -0.0743476003408432, -0.7412710189819336, 0.4620560109615326, -0.5836250185966492, -0.5901830196380615, 0.6233999729156494, 0.535269021987915, -0.6708179712295532, -0.9556419849395752, -0.7701730132102966, 0.4799860119819641, 0.664376974105835, 0.3994449973106384, -0.9688739776611328, -0.2762629985809326, -0.901951014995575, 0.5441039800643921, -0.9589809775352478, 0.4826579988002777, -0.8072839975357056, 0.30536898970603943, -0.9478179812431335, 0.8274980187416077, -0.38288700580596924, -0.805741012096405, -0.7966780066490173, -0.2998040020465851, -0.22982800006866455, 0.8187829852104187, -0.10305500030517578, -0.4556800127029419, -0.22782699763774872, 0.5437430143356323, -0.9607300162315369, 0.9467470049858093, -0.8571820259094238, -0.9642599821090698, -0.2924109995365143, -0.7156140208244324, 0.7652779817581177, -0.47504299879074097, -0.590142011642456, -0.2385070025920868, 0.6730020046234131, -0.4733569920063019, -0.31962600350379944, 0.9360139966011047, 0.48660698533058167, 0.580843985080719, 0.42535200715065, -0.8009939789772034, 0.2907629907131195, -0.4949530065059662, -0.44116199016571045, 0.7186769843101501, -0.8284270167350769, 0.9696499705314636, 7.536369957961142e-05, -0.6999729871749878, -0.526885986328125, -0.3526819944381714, 0.7994660139083862, 0.33278900384902954, 0.7233890295028687, 0.40765899419784546, -0.9340839982032776, -0.2847050130367279, 0.9614840149879456, -0.7003949880599976, -0.9858080148696899, -0.5953419804573059, -0.6917210221290588, 0.4944800138473511, -0.08426489681005478, 0.03909660130739212, 0.29893800616264343, -0.12809400260448456, -0.9715800285339355, 0.8639299869537354, 0.27060601115226746, -0.46898600459098816, -0.25660499930381775, 0.4721499979496002, -0.2731170058250427, -0.5903429985046387, -0.8265290260314941, -0.7253810167312622, -0.19482100009918213, -0.2596609890460968, -0.09492070227861404, -0.1803019940853119, 0.04468340054154396, -0.22213299572467804, -0.40393000841140747, 0.295771986246109, -0.9294899702072144, 0.5800790190696716, -0.169855996966362, 0.33031100034713745, 0.017355099320411682, -0.6358230113983154, 0.4759419858455658, 0.9071750044822693, 0.2427770048379898, -0.5122079849243164, 0.36246299743652344, 0.04962889850139618, 0.6517099738121033, 0.9900569915771484, 0.690733015537262, -0.4690130054950714, -0.10131099820137024, -0.6837199926376343, -0.15784099698066711, -0.677711009979248, -0.7082239985466003, -0.6594370007514954, -0.40760698914527893, 0.677033007144928, 0.8903200030326843, 0.22830699384212494, -0.7495139837265015, 0.772957980632782, 0.054701000452041626, 0.551705002784729, 0.9170519709587097, -0.8950219750404358, -0.7023969888687134, 0.484142005443573, 0.10864800214767456, 0.8333470225334167, 0.47887200117111206, -0.984112024307251, 0.3871760070323944, -0.732990026473999, 0.7526000142097473, 0.44331198930740356, -0.09878560155630112, 0.12541499733924866, 0.10875999927520752, -0.49810799956321716, 0.43209001421928406, 0.34460899233818054, 0.928941011428833, -0.130731999874115, -0.056916698813438416];
    let op44_expect = [-0.0006142780184745789, -1.2122100591659546, 0.4438610076904297, 0.10211700201034546, -2.527139902114868, 1.4748899936676025, 0.1734739989042282, -0.23757700622081757, 1.287350058555603, 1.9131499528884888, 2.5173399448394775, 0.3758409917354584, 0.6375629901885986, 2.6530001163482666, 2.7295899391174316, -1.6270999908447266, 1.1738899946212769, -2.121190071105957, 2.914170026779175, -2.242460012435913, 0.049704499542713165, -0.12710699439048767, -0.14447300136089325, -0.13376200199127197, -0.39328399300575256, -2.0234599113464355, -0.23917800188064575, -0.24650800228118896, 1.2927700281143188, 1.3296300172805786, 0.1175210028886795, 1.2237199544906616, 0.06657130271196365, 1.0943800210952759, -1.3142600059509277, 2.52593994140625, -0.9692109823226929, 0.5154780149459839, -1.6092599630355835, -0.8389049768447876, 0.1352110058069229, 0.7864149808883667, -1.14382004737854, -0.7391020059585571, -1.0173100233078003, 0.2816149890422821, 2.363110065460205, 0.8918229937553406, 1.9387199878692627, -0.15049099922180176, 3.452169895172119, 2.2821900844573975, 1.1828199625015259, -2.2508599758148193, 3.054680109024048, 0.16622799634933472, 0.43455401062965393, -2.5752899646759033, -0.9586619734764099, -2.2397799491882324, 2.667759895324707, 0.5426009893417358, 1.7610700130462646, -1.081339955329895];

    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type80_length = product(type80.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type82 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
    let type82_length = product(type82.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type81);
    let op24 = operandIndex++;
    model.addOperand(type80);
    let op34 = operandIndex++;
    model.addOperand(type39);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op44 = operandIndex++;
    model.addOperand(type82);

    model.setOperandValue(op24, new Float32Array([-0.9662129878997803, -0.4674740135669708, -0.8220300078392029, -0.5794550180435181, 0.027880899608135223, -0.7994599938392639, -0.6842589974403381, 0.5632380247116089, 0.37288999557495117, 0.738215982913971, 0.38604500889778137, -0.9177749752998352, 0.18432499468326569, -0.27056801319122314, 0.8223599791526794, 0.09736829996109009, -0.9413080215454102, -0.14470599591732025]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param25, param26, param27, param28], [op44]);

    model.identifyInputsAndOutputs([op14], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op44_output = new Float32Array(type82_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type82_length; ++i) {
      assert.isTrue(almostEqualCTS(op44_output[i], op44_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-46', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op45_expect = [1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type13 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type13_length = product(type13.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-46', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op45_expect = [1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type13 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type13_length = product(type13.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-47', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op45_expect = [1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type13 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type13_length = product(type13.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-47', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op45_expect = [1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type11_length = product(type11.dimensions);
    let type13 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type13_length = product(type13.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op14 = operandIndex++;
    model.addOperand(type10);
    let op24 = operandIndex++;
    model.addOperand(type11);
    let op34 = operandIndex++;
    model.addOperand(type3);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op24, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-48', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.8699309825897217, 0.6446279883384705, -0.9183930158615112, 0.153671994805336, 0.8685619831085205, -0.3581770062446594, -0.13493099808692932, -0.24756500124931335, 0.2217400074005127, -0.2591570019721985, -0.2842960059642792, -0.5380650162696838, 0.7655590176582336, 0.41986000537872314, -0.556240975856781, 0.6584939956665039, 0.214355006814003, -0.8501690030097961, -0.25289300084114075, -0.47893500328063965, 0.5305259823799133, -0.07006630301475525, -0.9887290000915527, -0.30306100845336914, 0.150845006108284, 0.8299149870872498, 0.47634899616241455, 0.4065369963645935, -0.3553430140018463, 0.7571449875831604, -0.356361985206604, 0.8004819750785828, -0.7138609886169434, 0.21048299968242645, -0.6343029737472534, 0.7182360291481018, -0.7520380020141602, 0.45754700899124146, -0.5507689714431763, -0.551177978515625, 0.4467659890651703, -0.22746199369430542, 0.21634800732135773, -0.8528059720993042, -0.3514859974384308, 0.5590599775314331, -0.6684929728507996, -0.3034929931163788, -0.3637630045413971, -0.16283699870109558, 0.07010120153427124, 0.7560970187187195, -0.14226900041103363, 0.3297240138053894, -0.6563169956207275, -0.9980859756469727, -0.6529489755630493, -0.4031600058078766, -0.8936820030212402, 0.43274399638175964, 0.6123620271682739, -0.8695880174636841, -0.7132700085639954, -0.39809200167655945, -0.04235589876770973, 0.4365760087966919, -0.9252719879150391, 0.1765490025281906, 0.8229039907455444, 0.0968329980969429, -0.29680201411247253, -0.4271950125694275, 0.0316540002822876, -0.25447899103164673, 0.2449049949645996, 0.09482540190219879, 0.6437690258026123, -0.9039099812507629, 0.3526650071144104, -0.9011790156364441, 0.2661589980125427, -0.96806800365448, -0.615401029586792, -0.38897499442100525, 0.9390519857406616, -0.11628899723291397, 0.10752300173044205, -0.058271098881959915, 0.4351719915866852, 0.33467501401901245, 0.4597109854221344, 0.7174360156059265, 0.49662700295448303, -0.6801750063896179, -0.4150660037994385, 0.33984801173210144, 0.5060039758682251, -0.3378080129623413, -0.10721799731254578, -0.1724960058927536, 0.8706380128860474, 0.9318720102310181, -0.9538840055465698, 0.9030420184135437, 0.7600780129432678, 0.20972700417041779, -0.28538399934768677, -0.45513999462127686, 0.11319400370121002, 0.07566110044717789, 0.09244350343942642, -0.47286298871040344, 0.9606090188026428, -0.16038499772548676, -0.8394449949264526, 0.45709699392318726, 0.1633480042219162, 0.3448669910430908, -0.13161900639533997, 0.6887149810791016, -0.5408269762992859, 0.5712590217590332, -0.9558699727058411, 0.506164014339447, -0.1558389961719513, 0.07896210253238678, 0.756771981716156, -0.6620690226554871, 0.24290800094604492, 0.4608210027217865, 0.17787200212478638, -0.2898389995098114, -0.6406030058860779, 0.702597975730896, -0.5064060091972351, -0.568261981010437, -0.07137160003185272, 0.4137920141220093, 0.15967300534248352, -0.3052079975605011, 0.13381600379943848, -0.16025400161743164, 0.787322998046875, -0.7532439827919006, 0.600721001625061, 0.2631860077381134, -0.16238699853420258, 0.4779619872570038, -0.7029510140419006, -0.7310360074043274, -0.9394810199737549, -0.5245190262794495, 0.9340720176696777, -0.5116369724273682, -0.5034989714622498, 0.10623600333929062, -0.3236840069293976, 0.5344439744949341, -0.8437449932098389, 0.36417099833488464, 0.03703580051660538, -0.16880099475383759, -0.4045589864253998, -0.8141779899597168, 0.9174500107765198, -0.3342759907245636, 0.6692500114440918, -0.8012009859085083, 0.15651099383831024, -0.4279490113258362, 0.3791530132293701, 0.8185970187187195, -0.6499019861221313, 0.4270870089530945, -0.586014986038208, -0.5597890019416809, -0.8339229822158813, 0.0892409011721611, -0.6212509870529175, 0.2138260006904602, 0.46550899744033813, 0.47040000557899475, 0.38026100397109985, 0.4130670130252838, 0.1808219999074936, 0.17286600172519684, 0.5961400270462036, 0.8255749940872192, 0.6629160046577454, -0.704380989074707, -0.29763099551200867, 0.6977779865264893];
    let op45_expect = [1.7200299501419067, 1.5581599473953247, 0.6675459742546082, 2.2366299629211426, 0.06615159660577774, 0.29025399684906006, 0.770222008228302, -1.581969976425171, -0.8505949974060059, -0.48422399163246155, 0.9499670267105103, -0.5772629976272583, -0.8719490170478821, 2.341320037841797, -0.1359650045633316, -0.985713005065918, 0.8151469826698303, 1.0311399698257446, -1.4191499948501587, -0.515533983707428, -0.37363898754119873, -1.5060399770736694, 0.6731129884719849, 3.061389923095703, -0.38857799768447876, -1.7670700550079346, -0.3156670033931732, -1.0381499528884888, 0.4327870011329651, -1.4164299964904785, 1.1294399499893188, -0.17580600082874298, -0.8464149832725525, 1.4009499549865723, 0.7083200216293335, 2.195620059967041, -2.6126599311828613, -0.7053830027580261, 1.261240005493164, 1.4654500484466553, -2.357609987258911, 2.0449399948120117];

    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type80_length = product(type80.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type87 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type87_length = product(type87.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type81);
    let op24 = operandIndex++;
    model.addOperand(type80);
    let op34 = operandIndex++;
    model.addOperand(type39);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type87);

    model.setOperandValue(op24, new Float32Array([-0.9662129878997803, -0.4674740135669708, -0.8220300078392029, -0.5794550180435181, 0.027880899608135223, -0.7994599938392639, -0.6842589974403381, 0.5632380247116089, 0.37288999557495117, 0.738215982913971, 0.38604500889778137, -0.9177749752998352, 0.18432499468326569, -0.27056801319122314, 0.8223599791526794, 0.09736829996109009, -0.9413080215454102, -0.14470599591732025]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type87_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type87_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-48', async function() {
    // For 'Conv2d v1_2' example: examples_1_H3_W2_VALID_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op14_value = [-0.29533499479293823, -0.0038760099560022354, -0.5522509813308716, 0.16608400642871857, -0.28481999039649963, -0.15214300155639648, -0.719884991645813, -0.8693860173225403, -0.7455980181694031, 0.823947012424469, 0.4731830060482025, -0.33133700489997864, 0.18763099610805511, 0.04265709966421127, -0.8268970251083374, -0.7550849914550781, -0.4724529981613159, -0.023365600034594536, 0.04834359884262085, 0.9334179759025574, -0.961974024772644, 0.012578300200402737, 0.21974200010299683, 0.34260401129722595, -0.1516599953174591, 0.09349049627780914, 0.7832210063934326, 0.12966400384902954, 0.8388440012931824, -0.2713879942893982, 0.9245190024375916, 0.3428429961204529, 0.2744179964065552, 0.3508169949054718, 0.8416380286216736, -0.5439929962158203, -0.002833950100466609, -0.12846699357032776, -0.6829429864883423, -0.31911700963974, 0.846340000629425, 0.2830030024051666, 0.32864999771118164, 0.029375499114394188, -0.03356960043311119, 0.5912659764289856, -0.0743476003408432, -0.7412710189819336, 0.4620560109615326, -0.5836250185966492, -0.5901830196380615, 0.6233999729156494, 0.535269021987915, -0.6708179712295532, -0.9556419849395752, -0.7701730132102966, 0.4799860119819641, 0.664376974105835, 0.3994449973106384, -0.9688739776611328, -0.2762629985809326, -0.901951014995575, 0.5441039800643921, -0.9589809775352478, 0.4826579988002777, -0.8072839975357056, 0.30536898970603943, -0.9478179812431335, 0.8274980187416077, -0.38288700580596924, -0.805741012096405, -0.7966780066490173, -0.2998040020465851, -0.22982800006866455, 0.8187829852104187, -0.10305500030517578, -0.4556800127029419, -0.22782699763774872, 0.5437430143356323, -0.9607300162315369, 0.9467470049858093, -0.8571820259094238, -0.9642599821090698, -0.2924109995365143, -0.7156140208244324, 0.7652779817581177, -0.47504299879074097, -0.590142011642456, -0.2385070025920868, 0.6730020046234131, -0.4733569920063019, -0.31962600350379944, 0.9360139966011047, 0.48660698533058167, 0.580843985080719, 0.42535200715065, -0.8009939789772034, 0.2907629907131195, -0.4949530065059662, -0.44116199016571045, 0.7186769843101501, -0.8284270167350769, 0.9696499705314636, 7.536369957961142e-05, -0.6999729871749878, -0.526885986328125, -0.3526819944381714, 0.7994660139083862, 0.33278900384902954, 0.7233890295028687, 0.40765899419784546, -0.9340839982032776, -0.2847050130367279, 0.9614840149879456, -0.7003949880599976, -0.9858080148696899, -0.5953419804573059, -0.6917210221290588, 0.4944800138473511, -0.08426489681005478, 0.03909660130739212, 0.29893800616264343, -0.12809400260448456, -0.9715800285339355, 0.8639299869537354, 0.27060601115226746, -0.46898600459098816, -0.25660499930381775, 0.4721499979496002, -0.2731170058250427, -0.5903429985046387, -0.8265290260314941, -0.7253810167312622, -0.19482100009918213, -0.2596609890460968, -0.09492070227861404, -0.1803019940853119, 0.04468340054154396, -0.22213299572467804, -0.40393000841140747, 0.295771986246109, -0.9294899702072144, 0.5800790190696716, -0.169855996966362, 0.33031100034713745, 0.017355099320411682, -0.6358230113983154, 0.4759419858455658, 0.9071750044822693, 0.2427770048379898, -0.5122079849243164, 0.36246299743652344, 0.04962889850139618, 0.6517099738121033, 0.9900569915771484, 0.690733015537262, -0.4690130054950714, -0.10131099820137024, -0.6837199926376343, -0.15784099698066711, -0.677711009979248, -0.7082239985466003, -0.6594370007514954, -0.40760698914527893, 0.677033007144928, 0.8903200030326843, 0.22830699384212494, -0.7495139837265015, 0.772957980632782, 0.054701000452041626, 0.551705002784729, 0.9170519709587097, -0.8950219750404358, -0.7023969888687134, 0.484142005443573, 0.10864800214767456, 0.8333470225334167, 0.47887200117111206, -0.984112024307251, 0.3871760070323944, -0.732990026473999, 0.7526000142097473, 0.44331198930740356, -0.09878560155630112, 0.12541499733924866, 0.10875999927520752, -0.49810799956321716, 0.43209001421928406, 0.34460899233818054, 0.928941011428833, -0.130731999874115, -0.056916698813438416];
    let op45_expect = [1.287350058555603, 1.9131499528884888, 2.5173399448394775, 0.3758409917354584, 0.6375629901885986, 2.6530001163482666, 2.7295899391174316, 1.1738899946212769, -2.121190071105957, 2.914170026779175, -2.242460012435913, 0.049704499542713165, -0.12710699439048767, -0.14447300136089325, -0.39328399300575256, -2.0234599113464355, -0.23917800188064575, -0.24650800228118896, 1.2927700281143188, 1.3296300172805786, 0.1175210028886795, 0.06657130271196365, 1.0943800210952759, -1.3142600059509277, 2.52593994140625, -0.9692109823226929, 0.5154780149459839, -1.6092599630355835, 0.1352110058069229, 0.7864149808883667, -1.14382004737854, -0.7391020059585571, -1.0173100233078003, 0.2816149890422821, 2.363110065460205, 1.9387199878692627, -0.15049099922180176, 3.452169895172119, 2.2821900844573975, 1.1828199625015259, -2.2508599758148193, 3.054680109024048];

    let type39 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type80 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type80_length = product(type80.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type87 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
    let type87_length = product(type87.dimensions);

    let op14 = operandIndex++;
    model.addOperand(type81);
    let op24 = operandIndex++;
    model.addOperand(type80);
    let op34 = operandIndex++;
    model.addOperand(type39);
    let param29 = operandIndex++;
    model.addOperand(type4);
    let param30 = operandIndex++;
    model.addOperand(type4);
    let param31 = operandIndex++;
    model.addOperand(type4);
    let param32 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type87);

    model.setOperandValue(op24, new Float32Array([-0.9662129878997803, -0.4674740135669708, -0.8220300078392029, -0.5794550180435181, 0.027880899608135223, -0.7994599938392639, -0.6842589974403381, 0.5632380247116089, 0.37288999557495117, 0.738215982913971, 0.38604500889778137, -0.9177749752998352, 0.18432499468326569, -0.27056801319122314, 0.8223599791526794, 0.09736829996109009, -0.9413080215454102, -0.14470599591732025]));
    model.setOperandValue(op34, new Float32Array([0.0]));
    model.setOperandValue(param29, new Int32Array([2]));
    model.setOperandValue(param30, new Int32Array([1]));
    model.setOperandValue(param31, new Int32Array([1]));
    model.setOperandValue(param32, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op14, op24, op34, param29, param30, param31, param32], [op45]);

    model.identifyInputsAndOutputs([op14], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op14_input = new Float32Array(op14_value);
    execution.setInput(0, op14_input);
    let op45_output = new Float32Array(type87_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type87_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-49', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op46_expect = [-1.27853, 1.74987, -0.876718, 0.989692, 0.298548, 0.522103, -0.536896, -0.179382, -0.966914, 1.33708, 1.37042, -0.495494, 1.43859, -1.548, -0.430026, -0.662793, -0.0867897, -0.900658, -0.524396, 0.255731, -0.779081, 0.12666, 0.915651, -0.444765, -0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.952311, -0.35811, 0.403449, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, -0.126476, -0.185224, -0.114779, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, 0.318668, 0.893795, -0.0600559, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, 1.13613, -1.22109, 1.4649, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.407478, -0.343106, -0.0353232, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845042, 2.48496, 0.385019, -0.201145, 0.533332, -0.904872, -0.333518, -0.581063, -2.07065, 0.118687, -1.86708, -0.601987, 0.432037, 1.73923, 0.590007, 0.419788, 0.314198, 2.12817, 0.570793, -1.15998, -0.348587, -1.10231, -2.13091, 0.134467, -0.460382, 0.138338, 3.455, 0.679068, -0.190282, -0.0307461];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-49', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op46_expect = [0.78574, 0.0700466, -0.110245, 0.0141003, -0.621007, -0.979104, 1.24104, 0.580398, -0.512997, 0.900559, -0.683229, -1.0162, 1.0089, -0.0752488, 0.110969, 0.270558, 0.756819, -0.10753, -0.371484, 0.149005, 0.0973829, 0.155766, -0.476502, 0.259481, 1.06709, -1.16534, 1.52694, -0.797245, 0.802736, -0.997109, 2.2661, -1.45548, 2.15506, -1.33682, 1.15225, -3.09324, 0.943457, 0.885211, 0.987944, -0.345875, -0.114708, 1.7107, 0.104745, 0.828324, -2.49964, -0.453742, -0.288829, -0.0948694, -0.489415, 1.74889, -0.378257, -2.10237, 0.613022, -2.5225, -0.746785, 3.63816, -1.9287, 0.774279, -0.613917, -0.650011, 1.03753, -0.177923, 0.891815, -1.00373, 1.83859, -1.59239, -0.0662623, 0.218806, -1.088, 0.280837, 0.902901, -1.90127, 3.04734, -1.57302, 1.10881, -0.980369, -3.85305, -0.955859, 1.64909, 2.33573, 0.31144, -0.594375, 0.325747, -0.952566, -0.613449, 2.85073, 1.94692, 1.12977, 1.1351, -0.449652, 0.118765, -0.199547, 2.873, 1.35182, -1.85457, 1.22364, 1.38049, 2.38342, 0.882321, 1.03795, -0.321571, -2.60202, -1.6372, 1.09302, 0.461768, 1.8485, -0.158928, 4.28871, -0.437375, -1.5794, 1.59869, 0.0811864, 0.912054, 0.452176, 2.01812, 2.62907, 1.50304, -0.840276, -0.455854, -0.224913, 0.609824, -0.11105, 3.35635, 2.02386, 1.4687, -0.708365, -0.508992, -3.02602, -0.75725, 1.85277, 2.92817, -0.172997, -1.13279, -0.355636, -0.337669, -0.588752, 2.05759, 1.0651, 0.884758, -0.0712112, 3.81319, 0.771629, 0.949634, 0.0838967, -2.19264, 0.114521, 0.543556, -1.63197, -0.267442, 1.15701, -2.37862, 2.57646, 0.531208, 0.9499, -0.231441, 1.51461, 1.58888, 0.895931, -0.753084, 0.545251, 0.746903, 0.012994, -0.790398, -1.1055, 1.77789, 0.430923, 0.818241, -0.731412, 0.979546, -2.48707, -1.53658, -1.66798, -1.04585, -0.667911, 1.00299, -2.20339, 0.137826, -2.31281, 0.755535, 0.495396, 0.549629, 0.713128, 0.751369, 0.283996, -0.814532, 1.4866, 1.12105, 0.927998, 0.517938, -0.612661, -1.47756, -1.42422];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-50', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op46_expect = [-1.27853, 1.74987, -0.876718, 0.989692, 0.298548, 0.522103, -0.536896, -0.179382, -0.966914, 1.33708, 1.37042, -0.495494, 1.43859, -1.548, -0.430026, -0.662793, -0.0867897, -0.900658, -0.524396, 0.255731, -0.779081, 0.12666, 0.915651, -0.444765, -0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.952311, -0.35811, 0.403449, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, -0.126476, -0.185224, -0.114779, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, 0.318668, 0.893795, -0.0600559, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, 1.13613, -1.22109, 1.4649, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.407478, -0.343106, -0.0353232, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845042, 2.48496, 0.385019, -0.201145, 0.533332, -0.904872, -0.333518, -0.581063, -2.07065, 0.118687, -1.86708, -0.601987, 0.432037, 1.73923, 0.590007, 0.419788, 0.314198, 2.12817, 0.570793, -1.15998, -0.348587, -1.10231, -2.13091, 0.134467, -0.460382, 0.138338, 3.455, 0.679068, -0.190282, -0.0307461];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-50', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op46_expect = [0.78574, 0.0700466, -0.110245, 0.0141003, -0.621007, -0.979104, 1.24104, 0.580398, -0.512997, 0.900559, -0.683229, -1.0162, 1.0089, -0.0752488, 0.110969, 0.270558, 0.756819, -0.10753, -0.371484, 0.149005, 0.0973829, 0.155766, -0.476502, 0.259481, 1.06709, -1.16534, 1.52694, -0.797245, 0.802736, -0.997109, 2.2661, -1.45548, 2.15506, -1.33682, 1.15225, -3.09324, 0.943457, 0.885211, 0.987944, -0.345875, -0.114708, 1.7107, 0.104745, 0.828324, -2.49964, -0.453742, -0.288829, -0.0948694, -0.489415, 1.74889, -0.378257, -2.10237, 0.613022, -2.5225, -0.746785, 3.63816, -1.9287, 0.774279, -0.613917, -0.650011, 1.03753, -0.177923, 0.891815, -1.00373, 1.83859, -1.59239, -0.0662623, 0.218806, -1.088, 0.280837, 0.902901, -1.90127, 3.04734, -1.57302, 1.10881, -0.980369, -3.85305, -0.955859, 1.64909, 2.33573, 0.31144, -0.594375, 0.325747, -0.952566, -0.613449, 2.85073, 1.94692, 1.12977, 1.1351, -0.449652, 0.118765, -0.199547, 2.873, 1.35182, -1.85457, 1.22364, 1.38049, 2.38342, 0.882321, 1.03795, -0.321571, -2.60202, -1.6372, 1.09302, 0.461768, 1.8485, -0.158928, 4.28871, -0.437375, -1.5794, 1.59869, 0.0811864, 0.912054, 0.452176, 2.01812, 2.62907, 1.50304, -0.840276, -0.455854, -0.224913, 0.609824, -0.11105, 3.35635, 2.02386, 1.4687, -0.708365, -0.508992, -3.02602, -0.75725, 1.85277, 2.92817, -0.172997, -1.13279, -0.355636, -0.337669, -0.588752, 2.05759, 1.0651, 0.884758, -0.0712112, 3.81319, 0.771629, 0.949634, 0.0838967, -2.19264, 0.114521, 0.543556, -1.63197, -0.267442, 1.15701, -2.37862, 2.57646, 0.531208, 0.9499, -0.231441, 1.51461, 1.58888, 0.895931, -0.753084, 0.545251, 0.746903, 0.012994, -0.790398, -1.1055, 1.77789, 0.430923, 0.818241, -0.731412, 0.979546, -2.48707, -1.53658, -1.66798, -1.04585, -0.667911, 1.00299, -2.20339, 0.137826, -2.31281, 0.755535, 0.495396, 0.549629, 0.713128, 0.751369, 0.283996, -0.814532, 1.4866, 1.12105, 0.927998, 0.517938, -0.612661, -1.47756, -1.42422];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-51', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.8699309825897217, 0.6446279883384705, -0.9183930158615112, 0.153671994805336, 0.8685619831085205, -0.3581770062446594, -0.13493099808692932, -0.24756500124931335, 0.2217400074005127, -0.2591570019721985, -0.2842960059642792, -0.5380650162696838, 0.7655590176582336, 0.41986000537872314, -0.556240975856781, 0.6584939956665039, 0.214355006814003, -0.8501690030097961, -0.25289300084114075, -0.47893500328063965, 0.5305259823799133, -0.07006630301475525, -0.9887290000915527, -0.30306100845336914, 0.150845006108284, 0.8299149870872498, 0.47634899616241455, 0.4065369963645935, -0.3553430140018463, 0.7571449875831604, -0.356361985206604, 0.8004819750785828, -0.7138609886169434, 0.21048299968242645, -0.6343029737472534, 0.7182360291481018, -0.7520380020141602, 0.45754700899124146, -0.5507689714431763, -0.551177978515625, 0.4467659890651703, -0.22746199369430542, 0.21634800732135773, -0.8528059720993042, -0.3514859974384308, 0.5590599775314331, -0.6684929728507996, -0.3034929931163788, -0.3637630045413971, -0.16283699870109558, 0.07010120153427124, 0.7560970187187195, -0.14226900041103363, 0.3297240138053894, -0.6563169956207275, -0.9980859756469727, -0.6529489755630493, -0.4031600058078766, -0.8936820030212402, 0.43274399638175964, 0.6123620271682739, -0.8695880174636841, -0.7132700085639954, -0.39809200167655945, -0.04235589876770973, 0.4365760087966919, -0.9252719879150391, 0.1765490025281906, 0.8229039907455444, 0.0968329980969429, -0.29680201411247253, -0.4271950125694275, 0.0316540002822876, -0.25447899103164673, 0.2449049949645996, 0.09482540190219879, 0.6437690258026123, -0.9039099812507629, 0.3526650071144104, -0.9011790156364441, 0.2661589980125427, -0.96806800365448, -0.615401029586792, -0.38897499442100525, 0.9390519857406616, -0.11628899723291397, 0.10752300173044205, -0.058271098881959915, 0.4351719915866852, 0.33467501401901245, 0.4597109854221344, 0.7174360156059265, 0.49662700295448303, -0.6801750063896179, -0.4150660037994385, 0.33984801173210144, 0.5060039758682251, -0.3378080129623413, -0.10721799731254578, -0.1724960058927536, 0.8706380128860474, 0.9318720102310181, -0.9538840055465698, 0.9030420184135437, 0.7600780129432678, 0.20972700417041779, -0.28538399934768677, -0.45513999462127686, 0.11319400370121002, 0.07566110044717789, 0.09244350343942642, -0.47286298871040344, 0.9606090188026428, -0.16038499772548676, -0.8394449949264526, 0.45709699392318726, 0.1633480042219162, 0.3448669910430908, -0.13161900639533997, 0.6887149810791016, -0.5408269762992859, 0.5712590217590332, -0.9558699727058411, 0.506164014339447, -0.1558389961719513, 0.07896210253238678, 0.756771981716156, -0.6620690226554871, 0.24290800094604492, 0.4608210027217865, 0.17787200212478638, -0.2898389995098114, -0.6406030058860779, 0.702597975730896, -0.5064060091972351, -0.568261981010437, -0.07137160003185272, 0.4137920141220093, 0.15967300534248352, -0.3052079975605011, 0.13381600379943848, -0.16025400161743164, 0.787322998046875, -0.7532439827919006, 0.600721001625061, 0.2631860077381134, -0.16238699853420258, 0.4779619872570038, -0.7029510140419006, -0.7310360074043274, -0.9394810199737549, -0.5245190262794495, 0.9340720176696777, -0.5116369724273682, -0.5034989714622498, 0.10623600333929062, -0.3236840069293976, 0.5344439744949341, -0.8437449932098389, 0.36417099833488464, 0.03703580051660538, -0.16880099475383759, -0.4045589864253998, -0.8141779899597168, 0.9174500107765198, -0.3342759907245636, 0.6692500114440918, -0.8012009859085083, 0.15651099383831024, -0.4279490113258362, 0.3791530132293701, 0.8185970187187195, -0.6499019861221313, 0.4270870089530945, -0.586014986038208, -0.5597890019416809, -0.8339229822158813, 0.0892409011721611, -0.6212509870529175, 0.2138260006904602, 0.46550899744033813, 0.47040000557899475, 0.38026100397109985, 0.4130670130252838, 0.1808219999074936, 0.17286600172519684, 0.5961400270462036, 0.8255749940872192, 0.6629160046577454, -0.704380989074707, -0.29763099551200867, 0.6977779865264893];
    let op46_expect = [-1.2785300016403198, 1.7498699426651, -0.8767179846763611, 0.989691972732544, 0.29854801297187805, 0.5221030116081238, -0.5368959903717041, -0.17938199639320374, -0.9669139981269836, 1.3370800018310547, 1.370419979095459, -0.49549400806427, 1.4385900497436523, -1.5479999780654907, -0.43002599477767944, -0.662792980670929, -0.08678969740867615, -0.9006580114364624, -0.5243960022926331, 0.2557309865951538, -0.7790809869766235, 0.12666000425815582, 0.9156510233879089, -0.44476500153541565, -0.18684199452400208, -1.8730800151824951, 1.2113499641418457, -0.3850089907646179, 1.7203199863433838, -1.5603599548339844, -1.2305899858474731, 1.2369400262832642, 0.0020001500379294157, 0.3595220148563385, 1.6008399724960327, 0.434006005525589, -0.28294500708580017, 2.372920036315918, -1.2865300178527832, 0.08478370308876038, -0.35209301114082336, -2.396589994430542, 0.1492460072040558, 0.9203510284423828, -1.343459963798523, 0.9523109793663025, -0.3581100106239319, 0.40344899892807007, 0.4847959876060486, -1.1998900175094604, -0.6842979788780212, -1.4130100011825562, 0.10317700356245041, -0.3070389926433563, 1.1774100065231323, 2.589359998703003, -2.7623701095581055, -1.215649962425232, -1.0961899757385254, 1.1743199825286865, 0.5121430158615112, 0.7713789939880371, 0.39987900853157043, -0.05330929905176163, 0.2908639907836914, 0.9556300044059753, 1.1632800102233887, 1.8076800107955933, -1.5256400108337402, -0.1264760047197342, -0.1852239966392517, -0.11477900296449661, 1.2247999906539917, 0.23712700605392456, -0.21329699456691742, -0.619940996170044, 0.4979439973831177, -1.6868799924850464, 1.5931400060653687, -0.1273369938135147, 0.11141899973154068, 1.1371899843215942, 1.6853699684143066, -0.4796440005302429, 1.186079978942871, -2.527440071105957, 1.3413599729537964, 0.5482969880104065, -2.0838000774383545, 2.6458499431610107, -0.9933540225028992, 0.1282380074262619, 1.2609200477600098, 0.318668007850647, 0.8937950134277344, -0.06005590036511421, -0.6291260123252869, -0.9492290019989014, 2.258280038833618, -1.9609999656677246, 0.0058959899470210075, -0.18785400688648224, -1.0240299701690674, 0.39612099528312683, 1.3703999519348145, 3.9935500621795654, 0.4342209994792938, 0.2744640111923218, -0.5624380111694336, -0.9148709774017334, 0.5391290187835693, -0.9286869764328003, 0.834954023361206, 0.8441780209541321, -0.5660529732704163, -0.9573410153388977, 0.9333360195159912, 1.1361299753189087, -1.2210899591445923, 1.464900016784668, -0.41466599702835083, -0.4528209865093231, -0.7060059905052185, -1.7265700101852417, -0.7265740036964417, -0.09793619811534882, -0.4786689877510071, 1.7870299816131592, -0.6392880082130432, 1.4856499433517456, -0.1799039989709854, 1.0100300312042236, -0.31711798906326294, -0.6753870248794556, 1.909690022468567, -1.383430004119873, 0.69725501537323, -0.2922550141811371, 1.8163399696350098, 0.7178009748458862, 0.8624789714813232, -0.407478004693985, -0.3431060016155243, -0.03532319888472557, -0.48189300298690796, -0.13556499779224396, -2.9594099521636963, 0.24784600734710693, 2.677570104598999, -2.239989995956421, -0.5196729898452759, 0.25444701313972473, 0.4152829945087433, -1.0106500387191772, 0.5079110264778137, 0.9799259901046753, -0.18430399894714355, -0.0009504369809292257, -0.7343479990959167, -0.19668500125408173, -0.7132409811019897, 0.5949720144271851, 0.08450420200824738, 2.4849600791931152, 0.3850190043449402, -0.2011449933052063, 0.5333319902420044, -0.9048720002174377, -0.33351799845695496, -0.5810629725456238, -2.070650100708008, 0.11868699640035629, -1.8670799732208252, -0.6019870042800903, 0.43203699588775635, 1.7392300367355347, 0.5900070071220398, 0.4197880029678345, 0.31419798731803894, 2.1281700134277344, 0.5707929730415344, -1.159980058670044, -0.3485870063304901, -1.1023099422454834, -2.1309099197387695, 0.1344670057296753, -0.46038201451301575, 0.13833799958229065, 3.4549999237060547, 0.6790680289268494, -0.19028200209140778, -0.0307461004704237];

    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type90 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type90_length = product(type90.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type81);
    let op25 = operandIndex++;
    model.addOperand(type90);
    let op35 = operandIndex++;
    model.addOperand(type64);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type81);

    model.setOperandValue(op25, new Float32Array([-0.9662129878997803, -0.5794550180435181, -0.6842589974403381, 0.738215982913971, 0.18432499468326569, 0.09736829996109009, -0.17686299979686737, -0.2393600046634674, -0.0002334040036657825, 0.055546000599861145, -0.2326579988002777, -0.3164040148258209, -0.012903999537229538, 0.32070499658584595, -0.3266569972038269, -0.9196739792823792, 0.8680809736251831, -0.8246080279350281, -0.4674740135669708, 0.027880899608135223, 0.5632380247116089, 0.38604500889778137, -0.27056801319122314, -0.9413080215454102, -0.7792270183563232, -0.2614920139312744, -0.7748039960861206, -0.7966499924659729, 0.2247299998998642, -0.4143120050430298, 0.6858969926834106, -0.3277919888496399, 0.7739499807357788, -0.7145779728889465, -0.9723650217056274, 0.06960990279912949, -0.8220300078392029, -0.7994599938392639, 0.37288999557495117, -0.9177749752998352, 0.8223599791526794, -0.14470599591732025, -0.16718800365924835, 0.2680619955062866, 0.7026410102844238, -0.4122230112552643, 0.7557590007781982, 0.72154700756073, -0.43636998534202576, -0.2749049961566925, -0.2691650092601776, 0.16101999580860138, 0.8198570013046265, -0.31200799345970154]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type81_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type81_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-51', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_SAME_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.29533499479293823, -0.0038760099560022354, -0.5522509813308716, 0.16608400642871857, -0.28481999039649963, -0.15214300155639648, -0.719884991645813, -0.8693860173225403, -0.7455980181694031, 0.823947012424469, 0.4731830060482025, -0.33133700489997864, 0.18763099610805511, 0.04265709966421127, -0.8268970251083374, -0.7550849914550781, -0.4724529981613159, -0.023365600034594536, 0.04834359884262085, 0.9334179759025574, -0.961974024772644, 0.012578300200402737, 0.21974200010299683, 0.34260401129722595, -0.1516599953174591, 0.09349049627780914, 0.7832210063934326, 0.12966400384902954, 0.8388440012931824, -0.2713879942893982, 0.9245190024375916, 0.3428429961204529, 0.2744179964065552, 0.3508169949054718, 0.8416380286216736, -0.5439929962158203, -0.002833950100466609, -0.12846699357032776, -0.6829429864883423, -0.31911700963974, 0.846340000629425, 0.2830030024051666, 0.32864999771118164, 0.029375499114394188, -0.03356960043311119, 0.5912659764289856, -0.0743476003408432, -0.7412710189819336, 0.4620560109615326, -0.5836250185966492, -0.5901830196380615, 0.6233999729156494, 0.535269021987915, -0.6708179712295532, -0.9556419849395752, -0.7701730132102966, 0.4799860119819641, 0.664376974105835, 0.3994449973106384, -0.9688739776611328, -0.2762629985809326, -0.901951014995575, 0.5441039800643921, -0.9589809775352478, 0.4826579988002777, -0.8072839975357056, 0.30536898970603943, -0.9478179812431335, 0.8274980187416077, -0.38288700580596924, -0.805741012096405, -0.7966780066490173, -0.2998040020465851, -0.22982800006866455, 0.8187829852104187, -0.10305500030517578, -0.4556800127029419, -0.22782699763774872, 0.5437430143356323, -0.9607300162315369, 0.9467470049858093, -0.8571820259094238, -0.9642599821090698, -0.2924109995365143, -0.7156140208244324, 0.7652779817581177, -0.47504299879074097, -0.590142011642456, -0.2385070025920868, 0.6730020046234131, -0.4733569920063019, -0.31962600350379944, 0.9360139966011047, 0.48660698533058167, 0.580843985080719, 0.42535200715065, -0.8009939789772034, 0.2907629907131195, -0.4949530065059662, -0.44116199016571045, 0.7186769843101501, -0.8284270167350769, 0.9696499705314636, 7.536369957961142e-05, -0.6999729871749878, -0.526885986328125, -0.3526819944381714, 0.7994660139083862, 0.33278900384902954, 0.7233890295028687, 0.40765899419784546, -0.9340839982032776, -0.2847050130367279, 0.9614840149879456, -0.7003949880599976, -0.9858080148696899, -0.5953419804573059, -0.6917210221290588, 0.4944800138473511, -0.08426489681005478, 0.03909660130739212, 0.29893800616264343, -0.12809400260448456, -0.9715800285339355, 0.8639299869537354, 0.27060601115226746, -0.46898600459098816, -0.25660499930381775, 0.4721499979496002, -0.2731170058250427, -0.5903429985046387, -0.8265290260314941, -0.7253810167312622, -0.19482100009918213, -0.2596609890460968, -0.09492070227861404, -0.1803019940853119, 0.04468340054154396, -0.22213299572467804, -0.40393000841140747, 0.295771986246109, -0.9294899702072144, 0.5800790190696716, -0.169855996966362, 0.33031100034713745, 0.017355099320411682, -0.6358230113983154, 0.4759419858455658, 0.9071750044822693, 0.2427770048379898, -0.5122079849243164, 0.36246299743652344, 0.04962889850139618, 0.6517099738121033, 0.9900569915771484, 0.690733015537262, -0.4690130054950714, -0.10131099820137024, -0.6837199926376343, -0.15784099698066711, -0.677711009979248, -0.7082239985466003, -0.6594370007514954, -0.40760698914527893, 0.677033007144928, 0.8903200030326843, 0.22830699384212494, -0.7495139837265015, 0.772957980632782, 0.054701000452041626, 0.551705002784729, 0.9170519709587097, -0.8950219750404358, -0.7023969888687134, 0.484142005443573, 0.10864800214767456, 0.8333470225334167, 0.47887200117111206, -0.984112024307251, 0.3871760070323944, -0.732990026473999, 0.7526000142097473, 0.44331198930740356, -0.09878560155630112, 0.12541499733924866, 0.10875999927520752, -0.49810799956321716, 0.43209001421928406, 0.34460899233818054, 0.928941011428833, -0.130731999874115, -0.056916698813438416];
    let op46_expect = [0.7857400178909302, 0.07004660367965698, -0.1102449968457222, 0.014100300148129463, -0.6210070252418518, -0.9791039824485779, 1.2410399913787842, 0.5803980231285095, -0.5129969716072083, 0.9005590081214905, -0.6832290291786194, -1.0161999464035034, 1.0089000463485718, -0.07524880021810532, 0.11096899956464767, 0.2705579996109009, 0.7568190097808838, -0.10752999782562256, -0.37148401141166687, 0.1490049958229065, 0.09738290309906006, 0.15576599538326263, -0.476502001285553, 0.2594810128211975, 1.0670900344848633, -1.1653399467468262, 1.5269399881362915, -0.7972450256347656, 0.8027359843254089, -0.9971089959144592, 2.2660999298095703, -1.4554799795150757, 2.155060052871704, -1.3368200063705444, 1.152250051498413, -3.0932400226593018, 0.9434570074081421, 0.8852109909057617, 0.9879440069198608, -0.34587499499320984, -0.11470799893140793, 1.7107000350952148, 0.10474500060081482, 0.828324019908905, -2.4996399879455566, -0.45374199748039246, -0.2888289988040924, -0.09486939758062363, -0.4894149899482727, 1.7488900423049927, -0.37825700640678406, -2.102370023727417, 0.6130220293998718, -2.5225000381469727, -0.7467849850654602, 3.638159990310669, -1.9286999702453613, 0.774278998374939, -0.6139169931411743, -0.6500110030174255, 1.0375299453735352, -0.17792299389839172, 0.8918150067329407, -1.003730058670044, 1.8385900259017944, -1.5923899412155151, -0.06626229733228683, 0.21880599856376648, -1.0880000591278076, 0.2808369994163513, 0.9029009938240051, -1.9012700319290161, 3.047339916229248, -1.5730199813842773, 1.1088099479675293, -0.980368971824646, -3.8530499935150146, -0.9558590054512024, 1.649090051651001, 2.3357300758361816, 0.31143999099731445, -0.5943750143051147, 0.325747013092041, -0.9525660276412964, -0.613448977470398, 2.8507299423217773, 1.9469200372695923, 1.129770040512085, 1.13510000705719, -0.4496519863605499, 0.11876499652862549, -0.19954699277877808, 2.872999906539917, 1.3518199920654297, -1.8545700311660767, 1.223639965057373, 1.3804899454116821, 2.383419990539551, 0.8823210000991821, 1.037950038909912, -0.3215709924697876, -2.602020025253296, -1.6371999979019165, 1.093019962310791, 0.4617680013179779, 1.8485000133514404, -0.1589280068874359, 4.288710117340088, -0.437375009059906, -1.5793999433517456, 1.5986900329589844, 0.08118639886379242, 0.9120540022850037, 0.4521760046482086, 2.018120050430298, 2.6290700435638428, 1.5030399560928345, -0.8402760028839111, -0.45585399866104126, -0.22491300106048584, 0.609824001789093, -0.11105000227689743, 3.3563499450683594, 2.023859977722168, 1.4687000513076782, -0.7083650231361389, -0.5089920163154602, -3.026020050048828, -0.7572500109672546, 1.8527699708938599, 2.9281699657440186, -0.17299699783325195, -1.132789969444275, -0.35563600063323975, -0.3376689851284027, -0.5887519717216492, 2.0575900077819824, 1.0650999546051025, 0.8847579956054688, -0.07121119648218155, 3.81318998336792, 0.7716289758682251, 0.9496340155601501, 0.0838966965675354, -2.1926400661468506, 0.11452099680900574, 0.5435559749603271, -1.6319700479507446, -0.267441987991333, 1.1570099592208862, -2.378619909286499, 2.5764598846435547, 0.5312079787254333, 0.9498999714851379, -0.23144100606441498, 1.5146100521087646, 1.588879942893982, 0.8959310054779053, -0.7530840039253235, 0.5452510118484497, 0.7469030022621155, 0.012993999756872654, -0.7903980016708374, -1.1054999828338623, 1.7778899669647217, 0.4309230148792267, 0.8182410001754761, -0.7314119935035706, 0.9795460104942322, -2.487070083618164, -1.536579966545105, -1.6679799556732178, -1.0458500385284424, -0.6679109930992126, 1.0029900074005127, -2.203389883041382, 0.13782599568367004, -2.312809944152832, 0.7555350065231323, 0.4953959882259369, 0.5496289730072021, 0.7131279706954956, 0.7513689994812012, 0.28399598598480225, -0.8145319819450378, 1.4866000413894653, 1.1210500001907349, 0.9279980063438416, 0.5179380178451538, -0.6126610040664673, -1.477560043334961, -1.4242199659347534];

    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type90 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type90_length = product(type90.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type81);
    let op25 = operandIndex++;
    model.addOperand(type90);
    let op35 = operandIndex++;
    model.addOperand(type64);
    let param33 = operandIndex++;
    model.addOperand(type4);
    let param34 = operandIndex++;
    model.addOperand(type4);
    let param35 = operandIndex++;
    model.addOperand(type4);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type81);

    model.setOperandValue(op25, new Float32Array([-0.9662129878997803, -0.5794550180435181, -0.6842589974403381, 0.738215982913971, 0.18432499468326569, 0.09736829996109009, -0.17686299979686737, -0.2393600046634674, -0.0002334040036657825, 0.055546000599861145, -0.2326579988002777, -0.3164040148258209, -0.012903999537229538, 0.32070499658584595, -0.3266569972038269, -0.9196739792823792, 0.8680809736251831, -0.8246080279350281, -0.4674740135669708, 0.027880899608135223, 0.5632380247116089, 0.38604500889778137, -0.27056801319122314, -0.9413080215454102, -0.7792270183563232, -0.2614920139312744, -0.7748039960861206, -0.7966499924659729, 0.2247299998998642, -0.4143120050430298, 0.6858969926834106, -0.3277919888496399, 0.7739499807357788, -0.7145779728889465, -0.9723650217056274, 0.06960990279912949, -0.8220300078392029, -0.7994599938392639, 0.37288999557495117, -0.9177749752998352, 0.8223599791526794, -0.14470599591732025, -0.16718800365924835, 0.2680619955062866, 0.7026410102844238, -0.4122230112552643, 0.7557590007781982, 0.72154700756073, -0.43636998534202576, -0.2749049961566925, -0.2691650092601776, 0.16101999580860138, 0.8198570013046265, -0.31200799345970154]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param33, new Int32Array([1]));
    model.setOperandValue(param34, new Int32Array([1]));
    model.setOperandValue(param35, new Int32Array([1]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param33, param34, param35, param36], [op46]);

    model.identifyInputsAndOutputs([op15], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op46_output = new Float32Array(type81_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type81_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-52', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op47_expect = [-0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845044, 2.48496, 0.385019];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type15_length = product(type15.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type15_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-52', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op47_expect = [1.06709, -1.16534, 1.52694, -0.797245, 0.802736, -0.997109, 2.2661, -1.45548, 2.15506, -1.33682, 1.15225, -3.09324, 0.943457, 0.885211, 0.987944, -0.345875, -0.114708, 1.7107, 0.104745, 0.828324, -2.49964, -0.489415, 1.74889, -0.378257, -2.10237, 0.613022, -2.5225, -0.746785, 3.63816, -1.9287, 0.774279, -0.613917, -0.650011, 1.03753, -0.177923, 0.891815, -1.00373, 1.83859, -1.59239, -0.0662623, 0.218806, -1.088, 3.04734, -1.57302, 1.10881, -0.980369, -3.85305, -0.955859, 1.64909, 2.33573, 0.31144, -0.594375, 0.325747, -0.952566, -0.613449, 2.85073, 1.94692, 1.12977, 1.1351, -0.449652, 0.118765, -0.199547, 2.873, 1.38049, 2.38342, 0.882321, 1.03795, -0.321571, -2.60202, -1.6372, 1.09302, 0.461768, 1.8485, -0.158928, 4.28871, -0.437375, -1.5794, 1.59869, 0.0811864, 0.912054, 0.452176, 2.01812, 2.62907, 1.50304, 0.609824, -0.11105, 3.35635, 2.02386, 1.4687, -0.708365, -0.508992, -3.02602, -0.75725, 1.85277, 2.92817, -0.172997, -1.13279, -0.355636, -0.337669, -0.588752, 2.05759, 1.0651, 0.884758, -0.0712112, 3.81319, -2.19264, 0.114521, 0.543556, -1.63197, -0.267442, 1.15701, -2.37862, 2.57646, 0.531208, 0.9499, -0.231441, 1.51461, 1.58888, 0.895931, -0.753084, 0.545251, 0.746904, 0.0129939, -0.790398, -1.1055, 1.77789];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type15_length = product(type15.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type15_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-53', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op47_expect = [-0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845044, 2.48496, 0.385019];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type15_length = product(type15.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type15_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-53', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op47_expect = [1.06709, -1.16534, 1.52694, -0.797245, 0.802736, -0.997109, 2.2661, -1.45548, 2.15506, -1.33682, 1.15225, -3.09324, 0.943457, 0.885211, 0.987944, -0.345875, -0.114708, 1.7107, 0.104745, 0.828324, -2.49964, -0.489415, 1.74889, -0.378257, -2.10237, 0.613022, -2.5225, -0.746785, 3.63816, -1.9287, 0.774279, -0.613917, -0.650011, 1.03753, -0.177923, 0.891815, -1.00373, 1.83859, -1.59239, -0.0662623, 0.218806, -1.088, 3.04734, -1.57302, 1.10881, -0.980369, -3.85305, -0.955859, 1.64909, 2.33573, 0.31144, -0.594375, 0.325747, -0.952566, -0.613449, 2.85073, 1.94692, 1.12977, 1.1351, -0.449652, 0.118765, -0.199547, 2.873, 1.38049, 2.38342, 0.882321, 1.03795, -0.321571, -2.60202, -1.6372, 1.09302, 0.461768, 1.8485, -0.158928, 4.28871, -0.437375, -1.5794, 1.59869, 0.0811864, 0.912054, 0.452176, 2.01812, 2.62907, 1.50304, 0.609824, -0.11105, 3.35635, 2.02386, 1.4687, -0.708365, -0.508992, -3.02602, -0.75725, 1.85277, 2.92817, -0.172997, -1.13279, -0.355636, -0.337669, -0.588752, 2.05759, 1.0651, 0.884758, -0.0712112, 3.81319, -2.19264, 0.114521, 0.543556, -1.63197, -0.267442, 1.15701, -2.37862, 2.57646, 0.531208, 0.9499, -0.231441, 1.51461, 1.58888, 0.895931, -0.753084, 0.545251, 0.746904, 0.0129939, -0.790398, -1.1055, 1.77789];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type10_length = product(type10.dimensions);
    let type14 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type15_length = product(type15.dimensions);
    let type4 = {type: nn.INT32};
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type8_length = product(type8.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type14);
    let op35 = operandIndex++;
    model.addOperand(type8);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type15);

    model.setOperandValue(op25, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type15_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type15_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-54', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.8699309825897217, 0.6446279883384705, -0.9183930158615112, 0.153671994805336, 0.8685619831085205, -0.3581770062446594, -0.13493099808692932, -0.24756500124931335, 0.2217400074005127, -0.2591570019721985, -0.2842960059642792, -0.5380650162696838, 0.7655590176582336, 0.41986000537872314, -0.556240975856781, 0.6584939956665039, 0.214355006814003, -0.8501690030097961, -0.25289300084114075, -0.47893500328063965, 0.5305259823799133, -0.07006630301475525, -0.9887290000915527, -0.30306100845336914, 0.150845006108284, 0.8299149870872498, 0.47634899616241455, 0.4065369963645935, -0.3553430140018463, 0.7571449875831604, -0.356361985206604, 0.8004819750785828, -0.7138609886169434, 0.21048299968242645, -0.6343029737472534, 0.7182360291481018, -0.7520380020141602, 0.45754700899124146, -0.5507689714431763, -0.551177978515625, 0.4467659890651703, -0.22746199369430542, 0.21634800732135773, -0.8528059720993042, -0.3514859974384308, 0.5590599775314331, -0.6684929728507996, -0.3034929931163788, -0.3637630045413971, -0.16283699870109558, 0.07010120153427124, 0.7560970187187195, -0.14226900041103363, 0.3297240138053894, -0.6563169956207275, -0.9980859756469727, -0.6529489755630493, -0.4031600058078766, -0.8936820030212402, 0.43274399638175964, 0.6123620271682739, -0.8695880174636841, -0.7132700085639954, -0.39809200167655945, -0.04235589876770973, 0.4365760087966919, -0.9252719879150391, 0.1765490025281906, 0.8229039907455444, 0.0968329980969429, -0.29680201411247253, -0.4271950125694275, 0.0316540002822876, -0.25447899103164673, 0.2449049949645996, 0.09482540190219879, 0.6437690258026123, -0.9039099812507629, 0.3526650071144104, -0.9011790156364441, 0.2661589980125427, -0.96806800365448, -0.615401029586792, -0.38897499442100525, 0.9390519857406616, -0.11628899723291397, 0.10752300173044205, -0.058271098881959915, 0.4351719915866852, 0.33467501401901245, 0.4597109854221344, 0.7174360156059265, 0.49662700295448303, -0.6801750063896179, -0.4150660037994385, 0.33984801173210144, 0.5060039758682251, -0.3378080129623413, -0.10721799731254578, -0.1724960058927536, 0.8706380128860474, 0.9318720102310181, -0.9538840055465698, 0.9030420184135437, 0.7600780129432678, 0.20972700417041779, -0.28538399934768677, -0.45513999462127686, 0.11319400370121002, 0.07566110044717789, 0.09244350343942642, -0.47286298871040344, 0.9606090188026428, -0.16038499772548676, -0.8394449949264526, 0.45709699392318726, 0.1633480042219162, 0.3448669910430908, -0.13161900639533997, 0.6887149810791016, -0.5408269762992859, 0.5712590217590332, -0.9558699727058411, 0.506164014339447, -0.1558389961719513, 0.07896210253238678, 0.756771981716156, -0.6620690226554871, 0.24290800094604492, 0.4608210027217865, 0.17787200212478638, -0.2898389995098114, -0.6406030058860779, 0.702597975730896, -0.5064060091972351, -0.568261981010437, -0.07137160003185272, 0.4137920141220093, 0.15967300534248352, -0.3052079975605011, 0.13381600379943848, -0.16025400161743164, 0.787322998046875, -0.7532439827919006, 0.600721001625061, 0.2631860077381134, -0.16238699853420258, 0.4779619872570038, -0.7029510140419006, -0.7310360074043274, -0.9394810199737549, -0.5245190262794495, 0.9340720176696777, -0.5116369724273682, -0.5034989714622498, 0.10623600333929062, -0.3236840069293976, 0.5344439744949341, -0.8437449932098389, 0.36417099833488464, 0.03703580051660538, -0.16880099475383759, -0.4045589864253998, -0.8141779899597168, 0.9174500107765198, -0.3342759907245636, 0.6692500114440918, -0.8012009859085083, 0.15651099383831024, -0.4279490113258362, 0.3791530132293701, 0.8185970187187195, -0.6499019861221313, 0.4270870089530945, -0.586014986038208, -0.5597890019416809, -0.8339229822158813, 0.0892409011721611, -0.6212509870529175, 0.2138260006904602, 0.46550899744033813, 0.47040000557899475, 0.38026100397109985, 0.4130670130252838, 0.1808219999074936, 0.17286600172519684, 0.5961400270462036, 0.8255749940872192, 0.6629160046577454, -0.704380989074707, -0.29763099551200867, 0.6977779865264893];
    let op47_expect = [-0.18684199452400208, -1.8730800151824951, 1.2113499641418457, -0.3850089907646179, 1.7203199863433838, -1.5603599548339844, -1.2305899858474731, 1.2369400262832642, 0.0020001500379294157, 0.3595220148563385, 1.6008399724960327, 0.434006005525589, -0.28294500708580017, 2.372920036315918, -1.2865300178527832, 0.08478370308876038, -0.35209301114082336, -2.396589994430542, 0.1492460072040558, 0.9203510284423828, -1.343459963798523, 0.4847959876060486, -1.1998900175094604, -0.6842979788780212, -1.4130100011825562, 0.10317700356245041, -0.3070389926433563, 1.1774100065231323, 2.589359998703003, -2.7623701095581055, -1.215649962425232, -1.0961899757385254, 1.1743199825286865, 0.5121430158615112, 0.7713789939880371, 0.39987900853157043, -0.05330929905176163, 0.2908639907836914, 0.9556300044059753, 1.1632800102233887, 1.8076800107955933, -1.5256400108337402, 1.2247999906539917, 0.23712700605392456, -0.21329699456691742, -0.619940996170044, 0.4979439973831177, -1.6868799924850464, 1.5931400060653687, -0.1273369938135147, 0.11141899973154068, 1.1371899843215942, 1.6853699684143066, -0.4796440005302429, 1.186079978942871, -2.527440071105957, 1.3413599729537964, 0.5482969880104065, -2.0838000774383545, 2.6458499431610107, -0.9933540225028992, 0.1282380074262619, 1.2609200477600098, -0.6291260123252869, -0.9492290019989014, 2.258280038833618, -1.9609999656677246, 0.0058959899470210075, -0.18785400688648224, -1.0240299701690674, 0.39612099528312683, 1.3703999519348145, 3.9935500621795654, 0.4342209994792938, 0.2744640111923218, -0.5624380111694336, -0.9148709774017334, 0.5391290187835693, -0.9286869764328003, 0.834954023361206, 0.8441780209541321, -0.5660529732704163, -0.9573410153388977, 0.9333360195159912, -0.41466599702835083, -0.4528209865093231, -0.7060059905052185, -1.7265700101852417, -0.7265740036964417, -0.09793619811534882, -0.4786689877510071, 1.7870299816131592, -0.6392880082130432, 1.4856499433517456, -0.1799039989709854, 1.0100300312042236, -0.31711798906326294, -0.6753870248794556, 1.909690022468567, -1.383430004119873, 0.69725501537323, -0.2922550141811371, 1.8163399696350098, 0.7178009748458862, 0.8624789714813232, -0.48189300298690796, -0.13556499779224396, -2.9594099521636963, 0.24784600734710693, 2.677570104598999, -2.239989995956421, -0.5196729898452759, 0.25444701313972473, 0.4152829945087433, -1.0106500387191772, 0.5079110264778137, 0.9799259901046753, -0.18430399894714355, -0.0009504369809292257, -0.7343479990959167, -0.19668500125408173, -0.7132409811019897, 0.5949720144271851, 0.08450440317392349, 2.4849600791931152, 0.3850190043449402];

    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type90 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type90_length = product(type90.dimensions);
    let type91 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type91_length = product(type91.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type81);
    let op25 = operandIndex++;
    model.addOperand(type90);
    let op35 = operandIndex++;
    model.addOperand(type64);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op25, new Float32Array([-0.9662129878997803, -0.5794550180435181, -0.6842589974403381, 0.738215982913971, 0.18432499468326569, 0.09736829996109009, -0.17686299979686737, -0.2393600046634674, -0.0002334040036657825, 0.055546000599861145, -0.2326579988002777, -0.3164040148258209, -0.012903999537229538, 0.32070499658584595, -0.3266569972038269, -0.9196739792823792, 0.8680809736251831, -0.8246080279350281, -0.4674740135669708, 0.027880899608135223, 0.5632380247116089, 0.38604500889778137, -0.27056801319122314, -0.9413080215454102, -0.7792270183563232, -0.2614920139312744, -0.7748039960861206, -0.7966499924659729, 0.2247299998998642, -0.4143120050430298, 0.6858969926834106, -0.3277919888496399, 0.7739499807357788, -0.7145779728889465, -0.9723650217056274, 0.06960990279912949, -0.8220300078392029, -0.7994599938392639, 0.37288999557495117, -0.9177749752998352, 0.8223599791526794, -0.14470599591732025, -0.16718800365924835, 0.2680619955062866, 0.7026410102844238, -0.4122230112552643, 0.7557590007781982, 0.72154700756073, -0.43636998534202576, -0.2749049961566925, -0.2691650092601776, 0.16101999580860138, 0.8198570013046265, -0.31200799345970154]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type91_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });
  it('check result for Conv2d v1_2 example-54', async function() {
    // For 'Conv2d v1_2' example: examples_3_H3_W2_VALID_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-0.29533499479293823, -0.0038760099560022354, -0.5522509813308716, 0.16608400642871857, -0.28481999039649963, -0.15214300155639648, -0.719884991645813, -0.8693860173225403, -0.7455980181694031, 0.823947012424469, 0.4731830060482025, -0.33133700489997864, 0.18763099610805511, 0.04265709966421127, -0.8268970251083374, -0.7550849914550781, -0.4724529981613159, -0.023365600034594536, 0.04834359884262085, 0.9334179759025574, -0.961974024772644, 0.012578300200402737, 0.21974200010299683, 0.34260401129722595, -0.1516599953174591, 0.09349049627780914, 0.7832210063934326, 0.12966400384902954, 0.8388440012931824, -0.2713879942893982, 0.9245190024375916, 0.3428429961204529, 0.2744179964065552, 0.3508169949054718, 0.8416380286216736, -0.5439929962158203, -0.002833950100466609, -0.12846699357032776, -0.6829429864883423, -0.31911700963974, 0.846340000629425, 0.2830030024051666, 0.32864999771118164, 0.029375499114394188, -0.03356960043311119, 0.5912659764289856, -0.0743476003408432, -0.7412710189819336, 0.4620560109615326, -0.5836250185966492, -0.5901830196380615, 0.6233999729156494, 0.535269021987915, -0.6708179712295532, -0.9556419849395752, -0.7701730132102966, 0.4799860119819641, 0.664376974105835, 0.3994449973106384, -0.9688739776611328, -0.2762629985809326, -0.901951014995575, 0.5441039800643921, -0.9589809775352478, 0.4826579988002777, -0.8072839975357056, 0.30536898970603943, -0.9478179812431335, 0.8274980187416077, -0.38288700580596924, -0.805741012096405, -0.7966780066490173, -0.2998040020465851, -0.22982800006866455, 0.8187829852104187, -0.10305500030517578, -0.4556800127029419, -0.22782699763774872, 0.5437430143356323, -0.9607300162315369, 0.9467470049858093, -0.8571820259094238, -0.9642599821090698, -0.2924109995365143, -0.7156140208244324, 0.7652779817581177, -0.47504299879074097, -0.590142011642456, -0.2385070025920868, 0.6730020046234131, -0.4733569920063019, -0.31962600350379944, 0.9360139966011047, 0.48660698533058167, 0.580843985080719, 0.42535200715065, -0.8009939789772034, 0.2907629907131195, -0.4949530065059662, -0.44116199016571045, 0.7186769843101501, -0.8284270167350769, 0.9696499705314636, 7.536369957961142e-05, -0.6999729871749878, -0.526885986328125, -0.3526819944381714, 0.7994660139083862, 0.33278900384902954, 0.7233890295028687, 0.40765899419784546, -0.9340839982032776, -0.2847050130367279, 0.9614840149879456, -0.7003949880599976, -0.9858080148696899, -0.5953419804573059, -0.6917210221290588, 0.4944800138473511, -0.08426489681005478, 0.03909660130739212, 0.29893800616264343, -0.12809400260448456, -0.9715800285339355, 0.8639299869537354, 0.27060601115226746, -0.46898600459098816, -0.25660499930381775, 0.4721499979496002, -0.2731170058250427, -0.5903429985046387, -0.8265290260314941, -0.7253810167312622, -0.19482100009918213, -0.2596609890460968, -0.09492070227861404, -0.1803019940853119, 0.04468340054154396, -0.22213299572467804, -0.40393000841140747, 0.295771986246109, -0.9294899702072144, 0.5800790190696716, -0.169855996966362, 0.33031100034713745, 0.017355099320411682, -0.6358230113983154, 0.4759419858455658, 0.9071750044822693, 0.2427770048379898, -0.5122079849243164, 0.36246299743652344, 0.04962889850139618, 0.6517099738121033, 0.9900569915771484, 0.690733015537262, -0.4690130054950714, -0.10131099820137024, -0.6837199926376343, -0.15784099698066711, -0.677711009979248, -0.7082239985466003, -0.6594370007514954, -0.40760698914527893, 0.677033007144928, 0.8903200030326843, 0.22830699384212494, -0.7495139837265015, 0.772957980632782, 0.054701000452041626, 0.551705002784729, 0.9170519709587097, -0.8950219750404358, -0.7023969888687134, 0.484142005443573, 0.10864800214767456, 0.8333470225334167, 0.47887200117111206, -0.984112024307251, 0.3871760070323944, -0.732990026473999, 0.7526000142097473, 0.44331198930740356, -0.09878560155630112, 0.12541499733924866, 0.10875999927520752, -0.49810799956321716, 0.43209001421928406, 0.34460899233818054, 0.928941011428833, -0.130731999874115, -0.056916698813438416];
    let op47_expect = [1.0670900344848633, -1.1653399467468262, 1.5269399881362915, -0.7972450256347656, 0.8027359843254089, -0.9971089959144592, 2.2660999298095703, -1.4554799795150757, 2.155060052871704, -1.3368200063705444, 1.152250051498413, -3.0932400226593018, 0.9434570074081421, 0.8852109909057617, 0.9879440069198608, -0.34587499499320984, -0.11470799893140793, 1.7107000350952148, 0.10474500060081482, 0.828324019908905, -2.4996399879455566, -0.4894149899482727, 1.7488900423049927, -0.37825700640678406, -2.102370023727417, 0.6130220293998718, -2.5225000381469727, -0.7467849850654602, 3.638159990310669, -1.9286999702453613, 0.774278998374939, -0.6139169931411743, -0.6500110030174255, 1.0375299453735352, -0.17792299389839172, 0.8918150067329407, -1.003730058670044, 1.8385900259017944, -1.5923899412155151, -0.06626229733228683, 0.21880599856376648, -1.0880000591278076, 3.047339916229248, -1.5730199813842773, 1.1088099479675293, -0.980368971824646, -3.8530499935150146, -0.9558590054512024, 1.649090051651001, 2.3357300758361816, 0.31143999099731445, -0.5943750143051147, 0.325747013092041, -0.9525660276412964, -0.613448977470398, 2.8507299423217773, 1.9469200372695923, 1.129770040512085, 1.13510000705719, -0.4496519863605499, 0.11876499652862549, -0.19954699277877808, 2.872999906539917, 1.3804899454116821, 2.383419990539551, 0.8823210000991821, 1.037950038909912, -0.3215709924697876, -2.602020025253296, -1.6371999979019165, 1.093019962310791, 0.4617680013179779, 1.8485000133514404, -0.1589280068874359, 4.288710117340088, -0.437375009059906, -1.5793999433517456, 1.5986900329589844, 0.08118639886379242, 0.9120540022850037, 0.4521760046482086, 2.018120050430298, 2.6290700435638428, 1.5030399560928345, 0.609824001789093, -0.11105000227689743, 3.3563499450683594, 2.023859977722168, 1.4687000513076782, -0.7083650231361389, -0.5089920163154602, -3.026020050048828, -0.7572500109672546, 1.8527699708938599, 2.9281699657440186, -0.17299699783325195, -1.132789969444275, -0.35563600063323975, -0.3376689851284027, -0.5887519717216492, 2.0575900077819824, 1.0650999546051025, 0.8847579956054688, -0.07121119648218155, 3.81318998336792, -2.1926400661468506, 0.11452099680900574, 0.5435559749603271, -1.6319700479507446, -0.267441987991333, 1.1570099592208862, -2.378619909286499, 2.5764598846435547, 0.5312079787254333, 0.9498999714851379, -0.23144100606441498, 1.5146100521087646, 1.588879942893982, 0.8959310054779053, -0.7530840039253235, 0.5452510118484497, 0.7469040155410767, 0.01299390010535717, -0.7903980016708374, -1.1054999828338623, 1.7778899669647217];

    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type64_length = product(type64.dimensions);
    let type81 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type81_length = product(type81.dimensions);
    let type90 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type90_length = product(type90.dimensions);
    let type91 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
    let type91_length = product(type91.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type81);
    let op25 = operandIndex++;
    model.addOperand(type90);
    let op35 = operandIndex++;
    model.addOperand(type64);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op25, new Float32Array([-0.9662129878997803, -0.5794550180435181, -0.6842589974403381, 0.738215982913971, 0.18432499468326569, 0.09736829996109009, -0.17686299979686737, -0.2393600046634674, -0.0002334040036657825, 0.055546000599861145, -0.2326579988002777, -0.3164040148258209, -0.012903999537229538, 0.32070499658584595, -0.3266569972038269, -0.9196739792823792, 0.8680809736251831, -0.8246080279350281, -0.4674740135669708, 0.027880899608135223, 0.5632380247116089, 0.38604500889778137, -0.27056801319122314, -0.9413080215454102, -0.7792270183563232, -0.2614920139312744, -0.7748039960861206, -0.7966499924659729, 0.2247299998998642, -0.4143120050430298, 0.6858969926834106, -0.3277919888496399, 0.7739499807357788, -0.7145779728889465, -0.9723650217056274, 0.06960990279912949, -0.8220300078392029, -0.7994599938392639, 0.37288999557495117, -0.9177749752998352, 0.8223599791526794, -0.14470599591732025, -0.16718800365924835, 0.2680619955062866, 0.7026410102844238, -0.4122230112552643, 0.7557590007781982, 0.72154700756073, -0.43636998534202576, -0.2749049961566925, -0.2691650092601776, 0.16101999580860138, 0.8198570013046265, -0.31200799345970154]));
    model.setOperandValue(op35, new Float32Array([0.0, 0.0, 0.0]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([1]));
    model.setOperandValue(param39, new Int32Array([1]));
    model.setOperandValue(param40, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param37, param38, param39, param40], [op47]);

    model.identifyInputsAndOutputs([op15], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Float32Array(op15_value);
    execution.setInput(0, op15_input);
    let op47_output = new Float32Array(type91_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });
});
