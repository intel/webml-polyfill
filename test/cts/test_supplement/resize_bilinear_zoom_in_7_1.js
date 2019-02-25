describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Resize bilinear by align_corners(TRUE) zoom in example/7', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3,  4,  4,  6,  5,  8,  6, 10,
                      5,  6,  6,  8,  7, 10,  8, 12,
                      7,  8,  8, 10,  9, 12, 10, 14,
                      9, 10, 10, 12, 11, 14, 12, 16,
                      3,  4,  4,  6,  5,  8,  6, 10,
                      5,  6,  6,  8,  7, 10,  8, 12,
                      7,  8,  8, 10,  9, 12, 10, 14,
                      9, 10, 10, 12, 11, 14, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 4, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);
    let align_corners = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([4]));
    model.setOperandValue(width, new Int32Array([4]));
    model.setOperandValue(align_corners, new Int32Array([1]));
    model.addOperation(nn.RESIZE_BILINEAR, [op1, height, width, align_corners], [op2]);

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
