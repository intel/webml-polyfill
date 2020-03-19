// Generated file (from: depthwise_conv2d_per_channel.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv2d per channel example-1', async function() {
    // For 'Depthwise conv2d per channel' example: examples_same
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 16, 4, 32, 4, 64, 4, 128];
    let op4_expect = [8, 48];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: 0};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type2);
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
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array([2, 4, 2, 0, 2, 2, 2, 0]));
    model.setOperandValue(op3, new Int32Array([0, 0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
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
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-2', async function() {
    // For 'Depthwise conv2d per channel' example: examples_same_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [4, 16, 4, 32, 4, 64, 4, 128];
    let op2_value = [2, 4, 2, 0, 2, 2, 2, 0];
    let op3_value = [0, 0];
    let op4_expect = [8, 48];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type11_length = product(type11.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: 0};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type2);
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
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
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
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-3', async function() {
    // For 'Depthwise conv2d per channel' example: examples_different
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op41_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type6 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type6_length = product(type6.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type6);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op31 = operandIndex++;
    model.addOperand(type7);
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
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op21, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op31, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([2]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12, param13, param14, param15], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type8_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-4', async function() {
    // For 'Depthwise conv2d per channel' example: examples_different_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op21_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let op31_value = [4, 4, 4, 4];
    let op41_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type12 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type12_length = product(type12.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type5);
    let op21 = operandIndex++;
    model.addOperand(type12);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op31 = operandIndex++;
    model.addOperand(type7);
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
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([1]));
    model.setOperandValue(param14, new Int32Array([2]));
    model.setOperandValue(param15, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op11, op21, op31, param8, param9, param10, param11, param12, param13, param14, param15], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type8_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-5', async function() {
    // For 'Depthwise conv2d per channel' example: examples_layout_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op42_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type10 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type10_length = product(type10.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type5);
    let op22 = operandIndex++;
    model.addOperand(type10);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op32 = operandIndex++;
    model.addOperand(type7);
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
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op22, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op32, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([1]));
    model.setOperandValue(param21, new Int32Array([1]));
    model.setOperandValue(param22, new Int32Array([2]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param16, param17, param18, param19, param20, param21, param22, param23], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type8_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Depthwise conv2d per channel example-6', async function() {
    // For 'Depthwise conv2d per channel' example: examples_layout_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130, 129, 130];
    let op22_value = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    let op32_value = [4, 4, 4, 4];
    let op42_expect = [132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131, 132, 130, 134, 131];

    let type13 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type13_length = product(type13.dimensions);
    let type4 = {type: nn.INT32};
    let type5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 128};
    let type5_length = product(type5.dimensions);
    let type7 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 128};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type5);
    let op22 = operandIndex++;
    model.addOperand(type13);
    model.setOperandSymmPerChannelQuantParams(op22, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op32 = operandIndex++;
    model.addOperand(type7);
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
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(op22, new Int8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param16, new Int32Array([0]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([1]));
    model.setOperandValue(param21, new Int32Array([1]));
    model.setOperandValue(param22, new Int32Array([2]));
    model.setOperandValue(param23, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op12, op22, op32, param16, param17, param18, param19, param20, param21, param22, param23], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type8_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });
});
