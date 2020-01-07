// Generated file (from: conv_float_2_relaxed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv float 2 relaxed example', async function() {
    // For 'Conv float 2 relaxed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let op4_expect = [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 4, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad_same = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let act_relu = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([1, 4, 7, 2, 5, 8, 3, 6, 9]));
    model.setOperandValue(op3, new Float32Array([-200]));
    model.setOperandValue(pad_same, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.setOperandValue(act_relu, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad_same, stride, stride, act_relu], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type0_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
