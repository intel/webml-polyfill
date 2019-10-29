describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for ARGMAX with 2D tensor using axis 0 example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 2.0, 4.0, 3.0];
    let op2_value = 0;
    let op3_expect = [1, 1];

    let type1 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type2 = {type: nn.TENSOR_INT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);

    let op2_input = new Int32Array([op2_value]);
    model.setOperandValue(op2, op2_input);

    model.addOperation(nn.ARGMAX, [op1, op2], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Int32Array(type2_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
