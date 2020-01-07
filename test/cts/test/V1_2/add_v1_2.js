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

  it('check result for Add v1_2 example-2', async function() {
    // For 'Add v1_2' example: examples_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [1, 2];
    let op21_value = [1, 2, 3, 4];
    let op31_expect = [2, 4, 4, 6];

    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type3_length = product(type3.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type2);
    let op21 = operandIndex++;
    model.addOperand(type3);
    let act1 = operandIndex++;
    model.addOperand(type1);
    let op31 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op21, new Float32Array(op21_value));

    model.setOperandValue(act1, new Int32Array([0]));
    model.addOperation(nn.ADD, [op11, op21, act1], [op31]);

    model.identifyInputsAndOutputs([op11], [op31]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Float32Array(op11_value);
    execution.setInput(0, op11_input);
    let op31_output = new Float32Array(type3_length);
    execution.setOutput(0, op31_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op31_output[i], op31_expect[i]));
    }
  });
});
