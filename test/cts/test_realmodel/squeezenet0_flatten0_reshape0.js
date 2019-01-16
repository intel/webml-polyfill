describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for reshape by squeezenet0_flatten0_reshape0', async function() {
    this.timeout(60000);
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op3_expect;
    await fetch('./cts/test_realmodel/resources/squeezenet0_pool3_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_flatten0_reshape0').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1000]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1000]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(op2, new Int32Array([1000]));
    model.addOperation(nn.RESHAPE, [op1, op2], [op3]);

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
