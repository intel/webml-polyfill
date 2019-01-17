describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for max_pool_2d by squeezenet0_pool1_fwd', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let i0_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat1').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool1_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type2_length = product(type2.dimensions);
    
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(stride, new Int32Array([2]));
    model.setOperandValue(filter, new Int32Array([3]));
    model.setOperandValue(padding, new Int32Array([0]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);

    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);

    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
