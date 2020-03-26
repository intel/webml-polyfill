// Generated file (from: conv2d_per_channel.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv2d per channel example-1', async function() {
    // For 'Conv2d per channel' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [138, 138, 138, 138, 138, 138];
    let op4_expect = [137, 141, 145, 137, 141, 145, 137, 141, 145];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
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
    let op4 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op3, new Int32Array([4, 4, 4]));
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
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-2', async function() {
    // For 'Conv2d per channel' example: examples_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [138, 138, 138, 138, 138, 138];
    let op2_value = [1, 2, 1, 2, 1, 2];
    let op3_value = [4, 4, 4];
    let op4_expect = [137, 141, 145, 137, 141, 145, 137, 141, 145];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type18 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type18_length = product(type18.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type18);
    model.setOperandSymmPerChannelQuantParams(op2, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
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
    let op4_output = new Uint8Array(type3_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-3', async function() {
    // For 'Conv2d per channel' example: examples_layouts_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [138, 108, 138, 108, 138, 108];
    let op41_expect = [121, 118, 115, 121, 118, 115, 121, 118, 115];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type0);
    let op21 = operandIndex++;
    model.addOperand(type6);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op31 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type4);
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
    let op41 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op21, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op31, new Int32Array([4, 4, 4]));
    model.setOperandValue(param7, new Int32Array([0]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([1]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11, param12, param13], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type3_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d per channel example-4', async function() {
    // For 'Conv2d per channel' example: examples_layouts_nhwc_weight_as_input
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [138, 108, 138, 108, 138, 108];
    let op21_value = [1, 2, 1, 2, 1, 2];
    let op31_value = [4, 4, 4];
    let op41_expect = [121, 118, 115, 121, 118, 115, 121, 118, 115];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 128};
    let type0_length = product(type0.dimensions);
    let type19 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type19_length = product(type19.dimensions);
    let type2 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 128};
    let type3_length = product(type3.dimensions);
    let type4 = {type: nn.INT32};

    let op11 = operandIndex++;
    model.addOperand(type0);
    let op21 = operandIndex++;
    model.addOperand(type19);
    model.setOperandSymmPerChannelQuantParams(op21, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op31 = operandIndex++;
    model.addOperand(type2);
    let param7 = operandIndex++;
    model.addOperand(type4);
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
    let op41 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op21, new Int8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([0]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.setOperandValue(param9, new Int32Array([0]));
    model.setOperandValue(param10, new Int32Array([0]));
    model.setOperandValue(param11, new Int32Array([1]));
    model.setOperandValue(param12, new Int32Array([1]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10, param11, param12, param13], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type3_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op41_output[i], op41_expect[i]));
    }
  });
});
