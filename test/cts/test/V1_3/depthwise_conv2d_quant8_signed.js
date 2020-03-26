// Generated file (from: depthwise_conv2d_quant8_signed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv2d quant8 signed example-1', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_same
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [-124, -112, -124, -96, -124, -64, -124, 0];
    let op45_expect = [-120, -80];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 2], scale: 0.5, zeroPoint: -128};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 2]};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type12_length = product(type12.dimensions);
    let type13 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 1, 2], scale: 1.0, zeroPoint: -128};
    let type13_length = product(type13.dimensions);
    let type4 = {type: nn.INT32};

    let op15 = operandIndex++;
    model.addOperand(type10);
    let op25 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op25, {channelDim: 3, scales: new Float32Array([0.5, 0.5])});
    let op35 = operandIndex++;
    model.addOperand(type12);
    let param41 = operandIndex++;
    model.addOperand(type4);
    let param42 = operandIndex++;
    model.addOperand(type4);
    let param43 = operandIndex++;
    model.addOperand(type4);
    let param44 = operandIndex++;
    model.addOperand(type4);
    let param45 = operandIndex++;
    model.addOperand(type4);
    let param46 = operandIndex++;
    model.addOperand(type4);
    let param47 = operandIndex++;
    model.addOperand(type4);
    let param48 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type13);

    model.setOperandValue(op25, new Int8Array([2, 4, 2, 0, 2, 2, 2, 0]));
    model.setOperandValue(op35, new Int32Array([0, 0]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.setOperandValue(param42, new Int32Array([0]));
    model.setOperandValue(param43, new Int32Array([0]));
    model.setOperandValue(param44, new Int32Array([0]));
    model.setOperandValue(param45, new Int32Array([1]));
    model.setOperandValue(param46, new Int32Array([1]));
    model.setOperandValue(param47, new Int32Array([1]));
    model.setOperandValue(param48, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op15, op25, op35, param41, param42, param43, param44, param45, param46, param47, param48], [op45]);

    model.identifyInputsAndOutputs([op15], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Int8Array(op15_value);
    execution.setInput(0, op15_input);
    let op45_output = new Int8Array(type13_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type13_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Depthwise conv2d quant8 signed example-2', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_different
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op16_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
    let op46_expect = [4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3];

    let type14 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type14_length = product(type14.dimensions);
    let type15 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type15_length = product(type15.dimensions);
    let type16 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 0};
    let type17_length = product(type17.dimensions);
    let type4 = {type: nn.INT32};

    let op16 = operandIndex++;
    model.addOperand(type14);
    let op26 = operandIndex++;
    model.addOperand(type15);
    model.setOperandSymmPerChannelQuantParams(op26, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op36 = operandIndex++;
    model.addOperand(type16);
    let param49 = operandIndex++;
    model.addOperand(type4);
    let param50 = operandIndex++;
    model.addOperand(type4);
    let param51 = operandIndex++;
    model.addOperand(type4);
    let param52 = operandIndex++;
    model.addOperand(type4);
    let param53 = operandIndex++;
    model.addOperand(type4);
    let param54 = operandIndex++;
    model.addOperand(type4);
    let param55 = operandIndex++;
    model.addOperand(type4);
    let param56 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(op26, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op36, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param49, new Int32Array([0]));
    model.setOperandValue(param50, new Int32Array([0]));
    model.setOperandValue(param51, new Int32Array([0]));
    model.setOperandValue(param52, new Int32Array([0]));
    model.setOperandValue(param53, new Int32Array([1]));
    model.setOperandValue(param54, new Int32Array([1]));
    model.setOperandValue(param55, new Int32Array([2]));
    model.setOperandValue(param56, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op16, op26, op36, param49, param50, param51, param52, param53, param54, param55, param56], [op46]);

    model.identifyInputsAndOutputs([op16], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op16_input = new Int8Array(op16_value);
    execution.setInput(0, op16_input);
    let op46_output = new Int8Array(type17_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Depthwise conv2d quant8 signed example-3', async function() {
    // For 'Depthwise conv2d quant8 signed' example: examples_layout_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op17_value = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
    let op47_expect = [4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3, 4, 2, 6, 3];

    let type14 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 2], scale: 0.5, zeroPoint: 0};
    let type14_length = product(type14.dimensions);
    let type16 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type16_length = product(type16.dimensions);
    let type17 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 4], scale: 1.0, zeroPoint: 0};
    let type17_length = product(type17.dimensions);
    let type18 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 4]};
    let type18_length = product(type18.dimensions);
    let type4 = {type: nn.INT32};

    let op17 = operandIndex++;
    model.addOperand(type14);
    let op27 = operandIndex++;
    model.addOperand(type18);
    model.setOperandSymmPerChannelQuantParams(op27, {channelDim: 3, scales: new Float32Array([1.0, 0.5, 1.0, 0.5])});
    let op37 = operandIndex++;
    model.addOperand(type16);
    let param57 = operandIndex++;
    model.addOperand(type4);
    let param58 = operandIndex++;
    model.addOperand(type4);
    let param59 = operandIndex++;
    model.addOperand(type4);
    let param60 = operandIndex++;
    model.addOperand(type4);
    let param61 = operandIndex++;
    model.addOperand(type4);
    let param62 = operandIndex++;
    model.addOperand(type4);
    let param63 = operandIndex++;
    model.addOperand(type4);
    let param64 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type17);

    model.setOperandValue(op27, new Int8Array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]));
    model.setOperandValue(op37, new Int32Array([4, 4, 4, 4]));
    model.setOperandValue(param57, new Int32Array([0]));
    model.setOperandValue(param58, new Int32Array([0]));
    model.setOperandValue(param59, new Int32Array([0]));
    model.setOperandValue(param60, new Int32Array([0]));
    model.setOperandValue(param61, new Int32Array([1]));
    model.setOperandValue(param62, new Int32Array([1]));
    model.setOperandValue(param63, new Int32Array([2]));
    model.setOperandValue(param64, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op17, op27, op37, param57, param58, param59, param60, param61, param62, param63, param64], [op47]);

    model.identifyInputsAndOutputs([op17], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op17_input = new Int8Array(op17_value);
    execution.setInput(0, op17_input);
    let op47_output = new Int8Array(type17_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type17_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op47_output[i], op47_expect[i]));
    }
  });
});
