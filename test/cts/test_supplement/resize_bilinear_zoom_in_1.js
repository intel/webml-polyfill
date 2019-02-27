describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Resize bilinear with inputs (without align_corners) zoom in example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 3.6666667, 4, 5, 7, 8, 6, 8.6666667, 10, 9, 9.6666667, 10, 11, 13, 14, 12, 14.6666667, 16];
    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op2_output = new Float32Array(type1_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
