// Generated file (from: depthwise_conv2d_float_2_relaxed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Depthwise conv2d float 2 relaxed example', async function() {
    // For 'Depthwise conv2d float 2 relaxed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12];
    let op4_expect = [71, -34, 99, -20, 91, -26, 127, -4];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1, 4]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad_valid = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let channelMultiplier = operandIndex++;
    model.addOperand(type3);
    let act_none = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16]));
    model.setOperandValue(op3, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad_valid, new Int32Array([2]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.setOperandValue(channelMultiplier, new Int32Array([2]));
    model.setOperandValue(act_none, new Int32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op1, op2, op3, pad_valid, stride, stride, channelMultiplier, act_none], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type4_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
