describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  
  it('check result for ATROUS_DEPTHWISE_CONV_2D same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11.0, 3.0, 7.2, 10.6, 11.0, 3.0, 7.4, 10.9, 6.0, 2.0, 7.6, 4.0, 11.0, 3.0, 7.8, 11.5, 11.0, 3.0, 8.0, 11.8, 6.0, 2.0, 8.2, 4.0, 6.0, 2.0, 8.4, 12.4, 6.0, 2.0, 8.6, 12.7, 3.5, 2.0, 8.8, 4.0];
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 4]};
    let type4_length = product(type4.dimensions);
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let bias = operandIndex++;
    model.addOperand(type2);
    let pad = operandIndex++;
    model.addOperand(type3);
    let rate_w = operandIndex++;
    model.addOperand(type3);
    let rate_h = operandIndex++;
    model.addOperand(type3);
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);
    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);
    await execution.startCompute();
    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D same example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [3.5, 3.0, 3.0, 4.0, 6.0, 3.0, 3.0, 4.0, 3.5, 2.0, 3.0, 4.0, 6.0, 3.0, 3.0, 10.6, 11.0, 3.0, 7.2, 10.9, 6.0, 2.0, 7.4, 4.0, 3.5, 2.0, 3.0, 11.5, 6.0, 2.0, 7.8, 11.8, 3.5, 2.0, 8.0, 4.0];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 4]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let bias = operandIndex++;
    model.addOperand(type2);
    let pad = operandIndex++;
    model.addOperand(type3);
    let rate_w = operandIndex++;
    model.addOperand(type3);
    let rate_h = operandIndex++;
    model.addOperand(type3);
    let mul = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, mul, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type4_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();
    console.log(op3_output);

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});