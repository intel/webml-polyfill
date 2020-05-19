// Generated file (from: avg_pool_quant8_signed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Avg pool quant8 signed example-1', async function() {
    // For 'Avg pool quant8 signed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-127, -126, -125, -124];
    let op3_expect = [-127, -126, -125, -124];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 0.5, zeroPoint: -128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let pad0 = operandIndex++;
    model.addOperand(type1);
    let cons1 = operandIndex++;
    model.addOperand(type1);
    let act = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(cons1, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Int8Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Int8Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Avg pool quant8 signed example-2', async function() {
    // For 'Avg pool quant8 signed' example: examples_4
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120];
    let op31_expect = [-128, -127, -126, -126, -126, -126, -126, -126, -126];

    let type1 = {type: nn.INT32};
    let type6 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: -128};
    let type6_length = product(type6.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type6);
    let pad01 = operandIndex++;
    model.addOperand(type1);
    let cons11 = operandIndex++;
    model.addOperand(type1);
    let relu1_activitation = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type6);

    model.setOperandValue(pad01, new Int32Array([0]));
    model.setOperandValue(cons11, new Int32Array([1]));
    model.setOperandValue(relu1_activitation, new Int32Array([2]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op11, pad01, pad01, pad01, pad01, cons11, cons11, cons11, cons11, relu1_activitation], [op31]);

    model.identifyInputsAndOutputs([op11], [op31]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Int8Array(op11_value);
    execution.setInput(0, op11_input);
    let op31_output = new Int8Array(type6_length);
    execution.setOutput(0, op31_output);

    await execution.startCompute();

    for (let i = 0; i < type6_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op31_output[i], op31_expect[i]));
    }
  });

  it('check result for Avg pool quant8 signed example-3', async function() {
    // For 'Avg pool quant8 signed' example: examples_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [-128, -32, -96, -64, -80, -96, 32, -16];
    let op32_expect = [-84, -36];

    let type1 = {type: nn.INT32};
    let type7 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 4, 1], scale: 0.0625, zeroPoint: -128};
    let type7_length = product(type7.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 2, 1], scale: 0.0625, zeroPoint: -128};
    let type8_length = product(type8.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type7);
    let pad_same = operandIndex++;
    model.addOperand(type1);
    let cons2 = operandIndex++;
    model.addOperand(type1);
    let act_none = operandIndex++;
    model.addOperand(type1);
    let op32 = operandIndex++;
    model.addOperand(type8);

    model.setOperandValue(pad_same, new Int32Array([1]));
    model.setOperandValue(cons2, new Int32Array([2]));
    model.setOperandValue(act_none, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op12, pad_same, cons2, cons2, cons2, cons2, act_none], [op32]);

    model.identifyInputsAndOutputs([op12], [op32]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Int8Array(op12_value);
    execution.setInput(0, op12_input);
    let op32_output = new Int8Array(type8_length);
    execution.setOutput(0, op32_output);

    await execution.startCompute();

    for (let i = 0; i < type8_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op32_output[i], op32_expect[i]));
    }
  });

  it('check result for Avg pool quant8 signed example-4', async function() {
    // For 'Avg pool quant8 signed' example: examples_nhwc_float32
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [-126, -124, -122, -120];
    let op4_expect = [-126, -124, -122, -120];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 2, 1], scale: 0.5, zeroPoint: -128};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};

    let op13 = operandIndex++;
    model.addOperand(type0);
    let param = operandIndex++;
    model.addOperand(type1);
    let param1 = operandIndex++;
    model.addOperand(type1);
    let param2 = operandIndex++;
    model.addOperand(type1);
    let param3 = operandIndex++;
    model.addOperand(type1);
    let param4 = operandIndex++;
    model.addOperand(type1);
    let param5 = operandIndex++;
    model.addOperand(type1);
    let param6 = operandIndex++;
    model.addOperand(type1);
    let param7 = operandIndex++;
    model.addOperand(type1);
    let param8 = operandIndex++;
    model.addOperand(type1);
    let op4 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([1]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op13, param, param1, param2, param3, param4, param5, param6, param7, param8], [op4]);

    model.identifyInputsAndOutputs([op13], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Int8Array(op13_value);
    execution.setInput(0, op13_input);
    let op4_output = new Int8Array(type0_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Avg pool quant8 signed example-5', async function() {
    // For 'Avg pool quant8 signed' example: examples_nhwc_float32_5
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op17_value = [-128, -104, -120, -112, -116, -120, -88, -100];
    let op44_expect = [-117, -105];

    let type1 = {type: nn.INT32};
    let type39 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 2, 4, 1], scale: 0.25, zeroPoint: -128};
    let type39_length = product(type39.dimensions);
    let type40 = {type: nn.TENSOR_QUANT8_ASYMM_SIGNED, dimensions: [1, 1, 2, 1], scale: 0.25, zeroPoint: -128};
    let type40_length = product(type40.dimensions);

    let op17 = operandIndex++;
    model.addOperand(type39);
    let param36 = operandIndex++;
    model.addOperand(type1);
    let param37 = operandIndex++;
    model.addOperand(type1);
    let param38 = operandIndex++;
    model.addOperand(type1);
    let param39 = operandIndex++;
    model.addOperand(type1);
    let param40 = operandIndex++;
    model.addOperand(type1);
    let param41 = operandIndex++;
    model.addOperand(type1);
    let op44 = operandIndex++;
    model.addOperand(type40);

    model.setOperandValue(param36, new Int32Array([1]));
    model.setOperandValue(param37, new Int32Array([2]));
    model.setOperandValue(param38, new Int32Array([2]));
    model.setOperandValue(param39, new Int32Array([2]));
    model.setOperandValue(param40, new Int32Array([2]));
    model.setOperandValue(param41, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op17, param36, param37, param38, param39, param40, param41], [op44]);

    model.identifyInputsAndOutputs([op17], [op44]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op17_input = new Int8Array(op17_value);
    execution.setInput(0, op17_input);
    let op44_output = new Int8Array(type40_length);
    execution.setOutput(0, op44_output);

    await execution.startCompute();

    for (let i = 0; i < type40_length; ++i) {
      assert.isTrue(almostEqualCTSQuant8(op44_output[i], op44_expect[i]));
    }
  });
});
