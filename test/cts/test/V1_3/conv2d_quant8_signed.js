// Generated file (from: conv2d_quant8_signed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv2d quant8 signed example-1', async function() {
    // For 'Conv2d quant8 signed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op15_value = [10, 10, 10, 10, 10, 10];
    let op45_expect = [9, 13, 17, 9, 13, 17, 9, 13, 17];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 0};
    let type10_length = product(type10.dimensions);
    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 0};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type9_length = product(type9.dimensions);

    let op15 = operandIndex++;
    model.addOperand(type7);
    let op25 = operandIndex++;
    model.addOperand(type8);
    model.setOperandSymmPerChannelQuantParams(op25, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op35 = operandIndex++;
    model.addOperand(type9);
    let param36 = operandIndex++;
    model.addOperand(type4);
    let param37 = operandIndex++;
    model.addOperand(type4);
    let param38 = operandIndex++;
    model.addOperand(type4);
    let param39 = operandIndex++;
    model.addOperand(type4);
    let param40 = operandIndex++;
    model.addOperand(type4);
    let param41 = operandIndex++;
    model.addOperand(type4);
    let param42 = operandIndex++;
    model.addOperand(type4);
    let op45 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op25, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op35, new Int32Array([4, 4, 4]));
    model.setOperandValue(param36, new Int32Array([0]));
    model.setOperandValue(param37, new Int32Array([0]));
    model.setOperandValue(param38, new Int32Array([0]));
    model.setOperandValue(param39, new Int32Array([0]));
    model.setOperandValue(param40, new Int32Array([1]));
    model.setOperandValue(param41, new Int32Array([1]));
    model.setOperandValue(param42, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op15, op25, op35, param36, param37, param38, param39, param40, param41, param42], [op45]);

    model.identifyInputsAndOutputs([op15], [op45]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op15_input = new Int8Array(op15_value);
    execution.setInput(0, op15_input);
    let op45_output = new Int8Array(type10_length);
    execution.setOutput(0, op45_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op45_output[i], op45_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-2', async function() {
    // For 'Conv2d quant8 signed' example: examples_layouts_nhwc
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op16_value = [10, -20, 10, -20, 10, -20];
    let op46_expect = [-7, -10, -13, -7, -10, -13, -7, -10, -13];

    let type10 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 3], scale: 1.0, zeroPoint: 0};
    let type10_length = product(type10.dimensions);
    let type11 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 2]};
    let type11_length = product(type11.dimensions);
    let type4 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 1, 2], scale: 0.5, zeroPoint: 0};
    let type7_length = product(type7.dimensions);
    let type9 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type9_length = product(type9.dimensions);

    let op16 = operandIndex++;
    model.addOperand(type7);
    let op26 = operandIndex++;
    model.addOperand(type11);
    model.setOperandSymmPerChannelQuantParams(op26, {channelDim: 0, scales: new Float32Array([0.5, 0.75, 1.0])});
    let op36 = operandIndex++;
    model.addOperand(type9);
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
    let param49 = operandIndex++;
    model.addOperand(type4);
    let op46 = operandIndex++;
    model.addOperand(type10);

    model.setOperandValue(op26, new Int8Array([1, 2, 1, 2, 1, 2]));
    model.setOperandValue(op36, new Int32Array([4, 4, 4]));
    model.setOperandValue(param43, new Int32Array([0]));
    model.setOperandValue(param44, new Int32Array([0]));
    model.setOperandValue(param45, new Int32Array([0]));
    model.setOperandValue(param46, new Int32Array([0]));
    model.setOperandValue(param47, new Int32Array([1]));
    model.setOperandValue(param48, new Int32Array([1]));
    model.setOperandValue(param49, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op16, op26, op36, param43, param44, param45, param46, param47, param48, param49], [op46]);

    model.identifyInputsAndOutputs([op16], [op46]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op16_input = new Int8Array(op16_value);
    execution.setInput(0, op16_input);
    let op46_output = new Int8Array(type10_length);
    execution.setOutput(0, op46_output);

    await execution.startCompute();

    for (let i = 0; i < type10_length; ++i) {
      assert.isTrue(almostEqualCTS(op46_output[i], op46_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-3', async function() {
    // For 'Conv2d quant8 signed' example: examples_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op17_value = [-126, -126, -126, -126, -127, -126, -126, -126, -126];
    let op47_expect = [-121, -121, -121, -121];

    let type4 = {type: nn.INT32};
    let type51 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: -128};
    let type51_length = product(type51.dimensions);
    let type54 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: -128};
    let type54_length = product(type54.dimensions);
    let type72 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 2, 2, 1]};
    let type72_length = product(type72.dimensions);
    let type73 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type73_length = product(type73.dimensions);

    let op17 = operandIndex++;
    model.addOperand(type51);
    let op27 = operandIndex++;
    model.addOperand(type72);
    model.setOperandSymmPerChannelQuantParams(op27, {channelDim: 0, scales: new Float32Array([0.125])});
    let op37 = operandIndex++;
    model.addOperand(type73);
    let param70 = operandIndex++;
    model.addOperand(type4);
    let param71 = operandIndex++;
    model.addOperand(type4);
    let param72 = operandIndex++;
    model.addOperand(type4);
    let param73 = operandIndex++;
    model.addOperand(type4);
    let param74 = operandIndex++;
    model.addOperand(type4);
    let param75 = operandIndex++;
    model.addOperand(type4);
    let param76 = operandIndex++;
    model.addOperand(type4);
    let op47 = operandIndex++;
    model.addOperand(type54);

    model.setOperandValue(op27, new Int8Array([2, 2, 2, 2]));
    model.setOperandValue(op37, new Int32Array([0]));
    model.setOperandValue(param70, new Int32Array([0]));
    model.setOperandValue(param71, new Int32Array([0]));
    model.setOperandValue(param72, new Int32Array([0]));
    model.setOperandValue(param73, new Int32Array([0]));
    model.setOperandValue(param74, new Int32Array([1]));
    model.setOperandValue(param75, new Int32Array([1]));
    model.setOperandValue(param76, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op17, op27, op37, param70, param71, param72, param73, param74, param75, param76], [op47]);

    model.identifyInputsAndOutputs([op17], [op47]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op17_input = new Int8Array(op17_value);
    execution.setInput(0, op17_input);
    let op47_output = new Int8Array(type54_length);
    execution.setOutput(0, op47_output);

    await execution.startCompute();

    for (let i = 0; i < type54_length; ++i) {
      assert.isTrue(almostEqualCTS(op47_output[i], op47_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-4', async function() {
    // For 'Conv2d quant8 signed' example: examples_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op18_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
    let op48_expect = [-78, -78, -78, -78, -43, 34, 79, -78, -78, -44, -17, -78];

    let type4 = {type: nn.INT32};
    let type74 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: -1};
    let type74_length = product(type74.dimensions);
    let type76 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: -78};
    let type76_length = product(type76.dimensions);
    let type77 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [1, 3, 3, 1]};
    let type77_length = product(type77.dimensions);
    let type78 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type78_length = product(type78.dimensions);

    let op18 = operandIndex++;
    model.addOperand(type74);
    let op28 = operandIndex++;
    model.addOperand(type77);
    model.setOperandSymmPerChannelQuantParams(op28, {channelDim: 0, scales: new Float32Array([0.5])});
    let op38 = operandIndex++;
    model.addOperand(type78);
    let param77 = operandIndex++;
    model.addOperand(type4);
    let param78 = operandIndex++;
    model.addOperand(type4);
    let param79 = operandIndex++;
    model.addOperand(type4);
    let param80 = operandIndex++;
    model.addOperand(type4);
    let op48 = operandIndex++;
    model.addOperand(type76);

    model.setOperandValue(op28, new Int8Array([2, 8, 14, 4, 10, 16, 6, 12, 18]));
    model.setOperandValue(op38, new Int32Array([-800]));
    model.setOperandValue(param77, new Int32Array([1]));
    model.setOperandValue(param78, new Int32Array([1]));
    model.setOperandValue(param79, new Int32Array([1]));
    model.setOperandValue(param80, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op18, op28, op38, param77, param78, param79, param80], [op48]);

    model.identifyInputsAndOutputs([op18], [op48]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op18_input = new Int8Array(op18_value);
    execution.setInput(0, op18_input);
    let op48_output = new Int8Array(type76_length);
    execution.setOutput(0, op48_output);

    await execution.startCompute();

    for (let i = 0; i < type76_length; ++i) {
      assert.isTrue(almostEqualCTS(op48_output[i], op48_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-5', async function() {
    // For 'Conv2d quant8 signed' example: examples_channel_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op19_value = [-118, -118, -118];
    let op49_expect = [-98, -53, -8];

    let type4 = {type: nn.INT32};
    let type45 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: -128};
    let type45_length = product(type45.dimensions);
    let type82 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type82_length = product(type82.dimensions);
    let type83 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type83_length = product(type83.dimensions);

    let op19 = operandIndex++;
    model.addOperand(type45);
    let op29 = operandIndex++;
    model.addOperand(type82);
    model.setOperandSymmPerChannelQuantParams(op29, {channelDim: 0, scales: new Float32Array([0.5, 0.4, 0.3])});
    let op39 = operandIndex++;
    model.addOperand(type83);
    let param81 = operandIndex++;
    model.addOperand(type4);
    let param82 = operandIndex++;
    model.addOperand(type4);
    let param83 = operandIndex++;
    model.addOperand(type4);
    let param84 = operandIndex++;
    model.addOperand(type4);
    let param85 = operandIndex++;
    model.addOperand(type4);
    let param86 = operandIndex++;
    model.addOperand(type4);
    let param87 = operandIndex++;
    model.addOperand(type4);
    let op49 = operandIndex++;
    model.addOperand(type45);

    model.setOperandValue(op29, new Int8Array([1, 2, 3, 5, 6, 8, 12, 13, 15]));
    model.setOperandValue(op39, new Int32Array([0, 0, 0]));
    model.setOperandValue(param81, new Int32Array([0]));
    model.setOperandValue(param82, new Int32Array([0]));
    model.setOperandValue(param83, new Int32Array([0]));
    model.setOperandValue(param84, new Int32Array([0]));
    model.setOperandValue(param85, new Int32Array([1]));
    model.setOperandValue(param86, new Int32Array([1]));
    model.setOperandValue(param87, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op19, op29, op39, param81, param82, param83, param84, param85, param86, param87], [op49]);

    model.identifyInputsAndOutputs([op19], [op49]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op19_input = new Int8Array(op19_value);
    execution.setInput(0, op19_input);
    let op49_output = new Int8Array(type45_length);
    execution.setOutput(0, op49_output);

    await execution.startCompute();

    for (let i = 0; i < type45_length; ++i) {
      assert.isTrue(almostEqualCTS(op49_output[i], op49_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-6', async function() {
    // For 'Conv2d quant8 signed' example: examples_large_nhwc_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op110_value = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36];
    let op410_expect = [-113, -110, -107, -95, -88, -80, -77, -65, -53, -59, -42, -26, -41, -20, 1, -23, 2, 28];

    let type4 = {type: nn.INT32};
    let type86 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 0};
    let type86_length = product(type86.dimensions);
    let type88 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: -128};
    let type88_length = product(type88.dimensions);
    let type89 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type89_length = product(type89.dimensions);
    let type90 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type90_length = product(type90.dimensions);

    let op110 = operandIndex++;
    model.addOperand(type86);
    let op210 = operandIndex++;
    model.addOperand(type89);
    model.setOperandSymmPerChannelQuantParams(op210, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 0.5])});
    let op310 = operandIndex++;
    model.addOperand(type90);
    let param88 = operandIndex++;
    model.addOperand(type4);
    let param89 = operandIndex++;
    model.addOperand(type4);
    let param90 = operandIndex++;
    model.addOperand(type4);
    let param91 = operandIndex++;
    model.addOperand(type4);
    let param92 = operandIndex++;
    model.addOperand(type4);
    let param93 = operandIndex++;
    model.addOperand(type4);
    let param94 = operandIndex++;
    model.addOperand(type4);
    let op410 = operandIndex++;
    model.addOperand(type88);

    model.setOperandValue(op210, new Int8Array([2, 8, 14, 2, 5, 8, 6, 12, 18]));
    model.setOperandValue(op310, new Int32Array([0, 0, 0]));
    model.setOperandValue(param88, new Int32Array([0]));
    model.setOperandValue(param89, new Int32Array([0]));
    model.setOperandValue(param90, new Int32Array([0]));
    model.setOperandValue(param91, new Int32Array([0]));
    model.setOperandValue(param92, new Int32Array([1]));
    model.setOperandValue(param93, new Int32Array([1]));
    model.setOperandValue(param94, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op110, op210, op310, param88, param89, param90, param91, param92, param93, param94], [op410]);

    model.identifyInputsAndOutputs([op110], [op410]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op110_input = new Int8Array(op110_value);
    execution.setInput(0, op110_input);
    let op410_output = new Int8Array(type88_length);
    execution.setOutput(0, op410_output);

    await execution.startCompute();

    for (let i = 0; i < type88_length; ++i) {
      assert.isTrue(almostEqualCTS(op410_output[i], op410_expect[i]));
    }
  });

  it('check result for Conv2d quant8 signed example-7', async function() {
    // For 'Conv2d quant8 signed' example: examples_large_nhwc_channelQuant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op110_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
    let op410_expect = [29, 35, 41, 65, 80, 95, 101, 125, 149, 137, 170, 203, 173, 215, 257, 209, 260, 311];

    let type4 = {type: nn.INT32};
    let type91 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 3, 3], scale: 1.0, zeroPoint: -1};
    let type91_length = product(type91.dimensions);
    let type92 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [3, 1, 1, 3]};
    let type92_length = product(type92.dimensions);
    let type93 = {type: nn.TENSOR_INT32, dimensions: [3]};
    let type93_length = product(type93.dimensions);

    let op110 = operandIndex++;
    model.addOperand(type91);
    let op210 = operandIndex++;
    model.addOperand(type92);
    model.setOperandSymmPerChannelQuantParams(op210, {channelDim: 0, scales: new Float32Array([0.5, 1.0, 1.005])});
    let op310 = operandIndex++;
    model.addOperand(type93);
    let param88 = operandIndex++;
    model.addOperand(type4);
    let param89 = operandIndex++;
    model.addOperand(type4);
    let param90 = operandIndex++;
    model.addOperand(type4);
    let param91 = operandIndex++;
    model.addOperand(type4);
    let param92 = operandIndex++;
    model.addOperand(type4);
    let param93 = operandIndex++;
    model.addOperand(type4);
    let param94 = operandIndex++;
    model.addOperand(type4);
    let op410 = operandIndex++;
    model.addOperand(type91);

    model.setOperandValue(op210, new Int8Array([2, 8, 14, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op310, new Int32Array([0, 0, 0]));
    model.setOperandValue(param88, new Int32Array([0]));
    model.setOperandValue(param89, new Int32Array([0]));
    model.setOperandValue(param90, new Int32Array([0]));
    model.setOperandValue(param91, new Int32Array([0]));
    model.setOperandValue(param92, new Int32Array([1]));
    model.setOperandValue(param93, new Int32Array([1]));
    model.setOperandValue(param94, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op110, op210, op310, param88, param89, param90, param91, param92, param93, param94], [op410]);

    model.identifyInputsAndOutputs([op110], [op410]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op110_input = new Int8Array(op110_value);
    execution.setInput(0, op110_input);
    let op410_output = new Int8Array(type91_length);
    execution.setOutput(0, op410_output);

    await execution.startCompute();

    for (let i = 0; i < type91_length; ++i) {
      assert.isTrue(almostEqualCTS(op410_output[i], op410_expect[i]));
    }
  });
});
