// Generated file (from: depthwise_conv2d_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv2d v1_2 example-1', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op4_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array([0.25, 0.0, 0.2, 0.0, 0.25, 0.0, 0.0, 0.3, 0.25, 0.0, 0.0, 0.0, 0.25, 0.1, 0.0, 0.0]));
    model.setOperandValue(op3, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

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

  it('check result for Depthwise conv2d v1_2 example-2', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op4_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Float32Array([0.25, 0.0, 0.2, 0.0, 0.25, 0.0, 0.0, 0.3, 0.25, 0.0, 0.0, 0.0, 0.25, 0.1, 0.0, 0.0]));
    model.setOperandValue(op3, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

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

  it('check result for Depthwise conv2d v1_2 example-3', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10.0, 21.0, 10.0, 22.0, 10.0, 23.0, 10.0, 24.0, 10.0, 25.0, 10.0, 26.0, 10.0, 27.0, 10.0, 28.0, 10.0, 29.0];
    let op4_expect = [11.0, 3.0, 7.199999809265137, 10.600000381469727, 11.0, 3.0, 7.400000095367432, 10.899999618530273, 11.0, 3.0, 7.800000190734863, 11.5, 11.0, 3.0, 8.0, 11.800000190734863];

    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type15_length = product(type15.dimensions);
    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type15);
    let op2 = operandIndex++;
    model.addOperand(type16);
    let op3 = operandIndex++;
    model.addOperand(type17);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type16);

    model.setOperandValue(op2, new Float32Array([0.25, 0.0, 0.20000000298023224, 0.0, 0.25, 0.0, 0.0, 0.30000001192092896, 0.25, 0.0, 0.0, 0.0, 0.25, 0.10000000149011612, 0.0, 0.0]));
    model.setOperandValue(op3, new Float32Array([1.0, 2.0, 3.0, 4.0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type16_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type16_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-4', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type19_length = product(type19.dimensions);
    let type20 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type20_length = product(type20.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type20);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Int8Array([25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0]));
    model.setOperandValue(op3, new Int32Array([200, 800, 600, 1600]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-5', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op4_expect = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type22 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type23_length = product(type23.dimensions);
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.0001, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type22);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type23);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type24);

    model.setOperandValue(op2, new Int8Array([25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0]));
    model.setOperandValue(op3, new Int32Array([200, 800, 600, 1600]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type24_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type24_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-6', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.01, zeroPoint: 0};
    let type25_length = product(type25.dimensions);
    let type26 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.005, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type25);
    let op3 = operandIndex++;
    model.addOperand(type26);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Uint8Array([25, 0, 20, 0, 25, 0, 0, 30, 25, 0, 0, 0, 25, 10, 0, 0]));
    model.setOperandValue(op3, new Int32Array([200, 400, 600, 800]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-7', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op2_value = [0.25, 0.0, 0.2, 0.0, 0.25, 0.0, 0.0, 0.3, 0.25, 0.0, 0.0, 0.0, 0.25, 0.1, 0.0, 0.0];
    let op3_value = [1, 2, 3, 4];
    let op4_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
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
    let param7 = operandIndex++;
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
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

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

  it('check result for Depthwise conv2d v1_2 example-8', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op2_value = [0.25, 0.0, 0.2, 0.0, 0.25, 0.0, 0.0, 0.3, 0.25, 0.0, 0.0, 0.0, 0.25, 0.1, 0.0, 0.0];
    let op3_value = [1, 2, 3, 4];
    let op4_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];

    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
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
    let param7 = operandIndex++;
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
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

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

  it('check result for Depthwise conv2d v1_2 example-9', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10.0, 21.0, 10.0, 22.0, 10.0, 23.0, 10.0, 24.0, 10.0, 25.0, 10.0, 26.0, 10.0, 27.0, 10.0, 28.0, 10.0, 29.0];
    let op2_value = [0.25, 0.0, 0.20000000298023224, 0.0, 0.25, 0.0, 0.0, 0.30000001192092896, 0.25, 0.0, 0.0, 0.0, 0.25, 0.10000000149011612, 0.0, 0.0];
    let op3_value = [1.0, 2.0, 3.0, 4.0];
    let op4_expect = [11.0, 3.0, 7.199999809265137, 10.600000381469727, 11.0, 3.0, 7.400000095367432, 10.899999618530273, 11.0, 3.0, 7.800000190734863, 11.5, 11.0, 3.0, 8.0, 11.800000190734863];

    let type15 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type15_length = product(type15.dimensions);
    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type15);
    let op2 = operandIndex++;
    model.addOperand(type16);
    let op3 = operandIndex++;
    model.addOperand(type17);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type16);

    model.setOperandValue(op2, new Float32Array(op2_value));
    model.setOperandValue(op3, new Float32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type16_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type16_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-10', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op2_value = [25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0];
    let op3_value = [200, 800, 600, 1600];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type19_length = product(type19.dimensions);
    let type20 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type20_length = product(type20.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type20);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-11', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op2_value = [25, 0, 20, 0, 25, 0, 0, 60, 25, 0, 0, 0, 25, 20, 0, 0];
    let op3_value = [200, 800, 600, 1600];
    let op4_expect = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type22 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type22_length = product(type22.dimensions);
    let type23 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type23_length = product(type23.dimensions);
    let type24 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.0001, zeroPoint: 0};
    let type24_length = product(type24.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type22);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.01, 0.005, 0.01, 0.005])});
    let op3 = operandIndex++;
    model.addOperand(type23);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type24);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type24_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type24_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-12', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [20, 42, 20, 44, 20, 46, 20, 48, 20, 50, 20, 52, 20, 54, 20, 56, 20, 58];
    let op2_value = [25, 0, 20, 0, 25, 0, 0, 30, 25, 0, 0, 0, 25, 10, 0, 0];
    let op3_value = [200, 400, 600, 800];
    let op4_expect = [110, 30, 72, 106, 110, 30, 74, 109, 110, 30, 78, 115, 110, 30, 80, 118];

    let type18 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type18_length = product(type18.dimensions);
    let type21 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.1, zeroPoint: 0};
    let type21_length = product(type21.dimensions);
    let type25 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.01, zeroPoint: 0};
    let type25_length = product(type25.dimensions);
    let type26 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.005, zeroPoint: 0};
    let type26_length = product(type26.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type18);
    let op2 = operandIndex++;
    model.addOperand(type25);
    let op3 = operandIndex++;
    model.addOperand(type26);
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
    let param7 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type21);

    model.setOperandValue(op2, new Uint8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([2]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6, param7], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type21_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type21_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-13', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12];
    let op41_expect = [71, -34, 99, -20, 91, -26, 127, -4];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type2);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op21, new Float32Array([1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16]));
    model.setOperandValue(op31, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

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

  it('check result for Depthwise conv2d v1_2 example-14', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12];
    let op41_expect = [71, -34, 99, -20, 91, -26, 127, -4];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type2);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op21, new Float32Array([1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16]));
    model.setOperandValue(op31, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

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

  it('check result for Depthwise conv2d v1_2 example-15', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 11.0, 12.0];
    let op41_expect = [71.0, -34.0, 99.0, -20.0, 91.0, -26.0, 127.0, -4.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type34 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type35_length = product(type35.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type34);
    let op21 = operandIndex++;
    model.addOperand(type16);
    let op31 = operandIndex++;
    model.addOperand(type17);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type35);

    model.setOperandValue(op21, new Float32Array([1.0, 2.0, 3.0, 4.0, -9.0, 10.0, -11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, -14.0, 15.0, -16.0]));
    model.setOperandValue(op31, new Float32Array([1.0, 2.0, 3.0, 4.0]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type35_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type35_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-16', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type38 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.25, zeroPoint: 0};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type37);
    let op31 = operandIndex++;
    model.addOperand(type38);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Uint8Array([130, 132, 134, 136, 110, 148, 106, 152, 138, 140, 142, 144, 154, 100, 158, 96]));
    model.setOperandValue(op31, new Int32Array([4, 8, 12, 16]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-17', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_channelQuant8_3
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
    let op31 = operandIndex++;
    model.addOperand(type41);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Int8Array([2, 8, 6, 16, -18, 40, -22, 48, 10, 24, 14, 32, 26, -56, 30, -64]));
    model.setOperandValue(op31, new Int32Array([4, 16, 12, 32]));
    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-18', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12];
    let op21_value = [1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16];
    let op31_value = [1, 2, 3, 4];
    let op41_expect = [71, -34, 99, -20, 91, -26, 127, -4];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type2);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

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

  it('check result for Depthwise conv2d v1_2 example-19', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12];
    let op21_value = [1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16];
    let op31_value = [1, 2, 3, 4];
    let op41_expect = [71, -34, 99, -20, 91, -26, 127, -4];

    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type2);
    let op31 = operandIndex++;
    model.addOperand(type3);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

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

  it('check result for Depthwise conv2d v1_2 example-20', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1.0, 2.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 5.0, 6.0, 11.0, 12.0];
    let op21_value = [1.0, 2.0, 3.0, 4.0, -9.0, 10.0, -11.0, 12.0, 5.0, 6.0, 7.0, 8.0, 13.0, -14.0, 15.0, -16.0];
    let op31_value = [1.0, 2.0, 3.0, 4.0];
    let op41_expect = [71.0, -34.0, 99.0, -20.0, 91.0, -26.0, 127.0, -4.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type34 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type34_length = product(type34.dimensions);
    let type35 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type35_length = product(type35.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type34);
    let op21 = operandIndex++;
    model.addOperand(type16);
    let op31 = operandIndex++;
    model.addOperand(type17);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type35);

    model.setOperandValue(op21, new Float32Array(op21_value));
    model.setOperandValue(op31, new Float32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Float32Array(type35_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type35_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-21', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op21_value = [130, 132, 134, 136, 110, 148, 106, 152, 138, 140, 142, 144, 154, 100, 158, 96];
    let op31_value = [4, 8, 12, 16];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type38 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.25, zeroPoint: 0};
    let type38_length = product(type38.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type37);
    let op31 = operandIndex++;
    model.addOperand(type38);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Uint8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-22', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_nhwc_weight_as_input_channelQuant8_3
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [130, 132, 142, 144, 134, 136, 146, 148, 138, 140, 150, 152];
    let op21_value = [2, 8, 6, 16, -18, 40, -22, 48, 10, 24, 14, 32, 26, -56, 30, -64];
    let op31_value = [4, 16, 12, 32];
    let op41_expect = [171, 66, 199, 80, 191, 74, 227, 96];

    let type36 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 2, 2], scale: 0.5, zeroPoint: 128};
    let type36_length = product(type36.dimensions);
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 1, 4], scale: 1.0, zeroPoint: 100};
    let type39_length = product(type39.dimensions);
    let type4 = {type: nn.INT32};
    let type40 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type40_length = product(type40.dimensions);
    let type41 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type41_length = product(type41.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type36);
    let op21 = operandIndex++;
    model.addOperand(type40);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([0.5, 0.25, 0.5, 0.25])});
    let op31 = operandIndex++;
    model.addOperand(type41);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type39);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([2]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.setOperandValue(param11, new Int32Array([2]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type39_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type39_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-23', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 21, 10, 22, 10, 23, 10, 24];
    let op42_expect = [110, 246];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type9_length = product(type9.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type7);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op22, new Float32Array([0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1]));
    model.setOperandValue(op32, new Float32Array([100, 200]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type9_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-24', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 21, 10, 22, 10, 23, 10, 24];
    let op42_expect = [110, 246];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type9_length = product(type9.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type7);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op22, new Float32Array([0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1]));
    model.setOperandValue(op32, new Float32Array([100, 200]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type9_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-25', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10.0, 21.0, 10.0, 22.0, 10.0, 23.0, 10.0, 24.0];
    let op42_expect = [110.0, 246.0];

    let type4 = {type: nn.INT32};
    let type48 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type50_length = product(type50.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type48);
    let op22 = operandIndex++;
    model.addOperand(type48);
    let op32 = operandIndex++;
    model.addOperand(type49);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type50);

    model.setOperandValue(op22, new Float32Array([0.25, 0.0, 0.25, 1.0, 0.25, 0.0, 0.25, 1.0]));
    model.setOperandValue(op32, new Float32Array([100.0, 200.0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type50_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type50_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-26', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [120, 142, 120, 144, 120, 146, 120, 148];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type51 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 100};
    let type51_length = product(type51.dimensions);
    let type52 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.125, zeroPoint: 128};
    let type52_length = product(type52.dimensions);
    let type53 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0625, zeroPoint: 0};
    let type53_length = product(type53.dimensions);
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type51);
    let op22 = operandIndex++;
    model.addOperand(type52);
    let op32 = operandIndex++;
    model.addOperand(type53);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Uint8Array([130, 128, 130, 136, 130, 128, 130, 136]));
    model.setOperandValue(op32, new Int32Array([1600, 3200]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-27', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [148, 170, 148, 172, 148, 174, 148, 176];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);
    let type55 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 128};
    let type55_length = product(type55.dimensions);
    let type56 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type56_length = product(type56.dimensions);
    let type57 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type57_length = product(type57.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type55);
    let op22 = operandIndex++;
    model.addOperand(type56);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
    let op32 = operandIndex++;
    model.addOperand(type57);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Int8Array([2, 0, 2, 4, 2, 0, 2, 4]));
    model.setOperandValue(op32, new Int32Array([1600, 1600]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-28', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 21, 10, 22, 10, 23, 10, 24];
    let op22_value = [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1];
    let op32_value = [100, 200];
    let op42_expect = [110, 246];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type9_length = product(type9.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type7);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type9_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-29', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 21, 10, 22, 10, 23, 10, 24];
    let op22_value = [0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1];
    let op32_value = [100, 200];
    let op42_expect = [110, 246];

    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type9_length = product(type9.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type7);
    let op22 = operandIndex++;
    model.addOperand(type7);
    let op32 = operandIndex++;
    model.addOperand(type8);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type9);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type9_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-30', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10.0, 21.0, 10.0, 22.0, 10.0, 23.0, 10.0, 24.0];
    let op22_value = [0.25, 0.0, 0.25, 1.0, 0.25, 0.0, 0.25, 1.0];
    let op32_value = [100.0, 200.0];
    let op42_expect = [110.0, 246.0];

    let type4 = {type: nn.INT32};
    let type48 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type49_length = product(type49.dimensions);
    let type50 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 2]};
    let type50_length = product(type50.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type48);
    let op22 = operandIndex++;
    model.addOperand(type48);
    let op32 = operandIndex++;
    model.addOperand(type49);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type50);

    model.setOperandValue(op22, new Float32Array(op22_value));
    model.setOperandValue(op32, new Float32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Float32Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Float32Array(type50_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type50_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-31', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [120, 142, 120, 144, 120, 146, 120, 148];
    let op22_value = [130, 128, 130, 136, 130, 128, 130, 136];
    let op32_value = [1600, 3200];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type51 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 100};
    let type51_length = product(type51.dimensions);
    let type52 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.125, zeroPoint: 128};
    let type52_length = product(type52.dimensions);
    let type53 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0625, zeroPoint: 0};
    let type53_length = product(type53.dimensions);
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type51);
    let op22 = operandIndex++;
    model.addOperand(type52);
    let op32 = operandIndex++;
    model.addOperand(type53);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Uint8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-32', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [148, 170, 148, 172, 148, 174, 148, 176];
    let op22_value = [2, 0, 2, 4, 2, 0, 2, 4];
    let op32_value = [1600, 1600];
    let op42_expect = [183, 251];

    let type4 = {type: nn.INT32};
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 2.0, zeroPoint: 128};
    let type54_length = product(type54.dimensions);
    let type55 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 128};
    let type55_length = product(type55.dimensions);
    let type56 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type56_length = product(type56.dimensions);
    let type57 = {type: nn.TENSOR_INT32, dimensions: [2], scale: 0.0, zeroPoint: 0};
    let type57_length = product(type57.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type55);
    let op22 = operandIndex++;
    model.addOperand(type56);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([0.125, 0.25])});
    let op32 = operandIndex++;
    model.addOperand(type57);
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
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([1]));
    model.setOperandValue(param18, new Int32Array([1]));
    model.setOperandValue(param19, new Int32Array([1]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param13, param14, param15, param16, param17, param18, param19, param20], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type54_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-33', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0];
    let op43_expect = [6010, 7046, 11000, 9000];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type10_length = product(type10.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op13 = operandIndex++;
    model.addOperand(type2);
    let op23 = operandIndex++;
    model.addOperand(type2);
    let op33 = operandIndex++;
    model.addOperand(type3);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op23, new Float32Array([0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50]));
    model.setOperandValue(op33, new Float32Array([6000, 7000, 8000, 9000]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type10_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-34', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0];
    let op43_expect = [6010, 7046, 11000, 9000];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type10_length = product(type10.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op13 = operandIndex++;
    model.addOperand(type2);
    let op23 = operandIndex++;
    model.addOperand(type2);
    let op33 = operandIndex++;
    model.addOperand(type3);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op23, new Float32Array([0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50]));
    model.setOperandValue(op33, new Float32Array([6000, 7000, 8000, 9000]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type10_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-35', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10.0, 21.0, 10.0, 0.0, 10.0, 22.0, 20.0, 0.0, 10.0, 23.0, 30.0, 0.0, 10.0, 24.0, 40.0, 0.0];
    let op43_expect = [6010.0, 7046.0, 11000.0, 9000.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};
    let type61 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type61_length = product(type61.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type16);
    let op23 = operandIndex++;
    model.addOperand(type16);
    let op33 = operandIndex++;
    model.addOperand(type17);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type61);

    model.setOperandValue(op23, new Float32Array([0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0]));
    model.setOperandValue(op33, new Float32Array([6000.0, 7000.0, 8000.0, 9000.0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type61_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type61_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-36', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type62 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.25, zeroPoint: 0};
    let type62_length = product(type62.dimensions);
    let type63 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.125, zeroPoint: 0};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type62);
    let op33 = operandIndex++;
    model.addOperand(type63);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Uint8Array([1, 0, 40, 200, 1, 4, 80, 200, 1, 0, 120, 200, 1, 4, 160, 200]));
    model.setOperandValue(op33, new Int32Array([48000, 56000, 64000, 72000]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-37', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);
    let type65 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type65_length = product(type65.dimensions);
    let type66 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type66_length = product(type66.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type65);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
    let op33 = operandIndex++;
    model.addOperand(type66);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Int8Array([0, 0, 10, 50, 0, 0, 20, 50, 0, 0, 30, 50, 0, 0, 40, 50]));
    model.setOperandValue(op33, new Int32Array([12000, 7000, 16000, 18000]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-38', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0];
    let op23_value = [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50];
    let op33_value = [6000, 7000, 8000, 9000];
    let op43_expect = [6010, 7046, 11000, 9000];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type10_length = product(type10.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op13 = operandIndex++;
    model.addOperand(type2);
    let op23 = operandIndex++;
    model.addOperand(type2);
    let op33 = operandIndex++;
    model.addOperand(type3);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type10_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-39', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_relaxed_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0];
    let op23_value = [0.25, 0, 10, 50, 0.25, 1, 20, 50, 0.25, 0, 30, 50, 0.25, 1, 40, 50];
    let op33_value = [6000, 7000, 8000, 9000];
    let op43_expect = [6010, 7046, 11000, 9000];

    let type10 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type10_length = product(type10.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op13 = operandIndex++;
    model.addOperand(type2);
    let op23 = operandIndex++;
    model.addOperand(type2);
    let op33 = operandIndex++;
    model.addOperand(type3);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type10_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-40', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_float16_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [10.0, 21.0, 10.0, 0.0, 10.0, 22.0, 20.0, 0.0, 10.0, 23.0, 30.0, 0.0, 10.0, 24.0, 40.0, 0.0];
    let op23_value = [0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0, 0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0];
    let op33_value = [6000.0, 7000.0, 8000.0, 9000.0];
    let op43_expect = [6010.0, 7046.0, 11000.0, 9000.0];

    let type16 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};
    let type61 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
    let type61_length = product(type61.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type16);
    let op23 = operandIndex++;
    model.addOperand(type16);
    let op33 = operandIndex++;
    model.addOperand(type17);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type61);

    model.setOperandValue(op23, new Float32Array(op23_value));
    model.setOperandValue(op33, new Float32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Float32Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Float32Array(type61_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type61_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-41', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op23_value = [1, 0, 40, 200, 1, 4, 80, 200, 1, 0, 120, 200, 1, 4, 160, 200];
    let op33_value = [48000, 56000, 64000, 72000];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type62 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.25, zeroPoint: 0};
    let type62_length = product(type62.dimensions);
    let type63 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.125, zeroPoint: 0};
    let type63_length = product(type63.dimensions);
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type62);
    let op33 = operandIndex++;
    model.addOperand(type63);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Uint8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Depthwise conv2d v1_2 example-42', async function() {
    // For 'Depthwise conv2d v1_2' example: examples_large_nhwc_weight_as_input_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [148, 170, 148, 128, 148, 172, 168, 128, 148, 174, 188, 128, 148, 176, 208, 128];
    let op23_value = [0, 0, 10, 50, 0, 0, 20, 50, 0, 0, 30, 50, 0, 0, 40, 50];
    let op33_value = [12000, 7000, 16000, 18000];
    let op43_expect = [120, 141, 220, 180];

    let type37 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 0.5, zeroPoint: 128};
    let type37_length = product(type37.dimensions);
    let type4 = {type: nn.INT32};
    let type64 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 4], scale: 50.0, zeroPoint: 0};
    let type64_length = product(type64.dimensions);
    let type65 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type65_length = product(type65.dimensions);
    let type66 = {type: nn.TENSOR_INT32, dimensions: [4], scale: 0.0, zeroPoint: 0};
    let type66_length = product(type66.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type37);
    let op23 = operandIndex++;
    model.addOperand(type65);
    model.setOperandSymmPerChannelQuantParams(op23, {channelDim: 3, scales: new Float32Array([1.0, 2.0, 1.0, 1.0])});
    let op33 = operandIndex++;
    model.addOperand(type66);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let param25 = operandIndex++;
    model.addOperand(type4);
    let param26 = operandIndex++;
    model.addOperand(type4);
    let param27 = operandIndex++;
    model.addOperand(type4);
    let param28 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type64);

    model.setOperandValue(op23, new Int8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([0]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.setOperandValue(param25, new Int32Array([1]));
    model.setOperandValue(param26, new Int32Array([1]));
    model.setOperandValue(param27, new Int32Array([1]));
    model.setOperandValue(param28, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op13, op23, op33, param21, param22, param23, param24, param25, param26, param27, param28], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type64_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type64_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op43_output[i], op43_expect[i]));
    }
  });
});
