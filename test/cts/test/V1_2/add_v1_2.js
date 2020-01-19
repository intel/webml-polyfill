// Generated file (from: add_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Add v1_2 example-1', async function() {
    // For 'Add v1_2' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0009765625, 1.0, 2.5];
    let op2_value = [2e-23, 0.0001, 3.5];
    let op3_expect = [1.0009765625, 1.0, 6.0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type0);
    let act = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array(op2_value));

    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.ADD, [op1, op2, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
