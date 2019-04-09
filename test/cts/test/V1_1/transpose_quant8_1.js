describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Transpose quant8 example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119];
    let output_expect = [0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104, 5, 6, 7, 8, 9, 25, 26, 27, 28, 29, 45, 46, 47, 48, 49, 65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109, 10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50, 51, 52, 53, 54, 70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114, 15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119];

    let type1 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 3, 4, 5], scale: 1.0, zeroPoint: 0};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 2, 3, 5], scale: 1.0, zeroPoint: 0};
    let type2_length = product(type2.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let perms = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(perms, new Int32Array([2, 0, 1, 3]));
    model.addOperation(nn.TRANSPOSE, [input, perms], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Uint8Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Uint8Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
