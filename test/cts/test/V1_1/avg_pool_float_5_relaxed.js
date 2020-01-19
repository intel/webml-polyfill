// Generated file (from: avg_pool_float_5_relaxed.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Avg pool float 5 relaxed example', async function() {
    // For 'Avg pool float 5 relaxed' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [0, 6, 2, 4, 3, 2, 10, 7];
    let op3_expect = [2.75, 5.75];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 4, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 1]};
    let type2_length = product(type2.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let pad_same = operandIndex++;
    model.addOperand(type1);
    let cons2 = operandIndex++;
    model.addOperand(type1);
    let act_none = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(pad_same, new Int32Array([1]));
    model.setOperandValue(cons2, new Int32Array([2]));
    model.setOperandValue(act_none, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [op1, pad_same, cons2, cons2, cons2, cons2, act_none], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type2_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
