describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  
  it('Check result for concatenation by squeezenet0_concat5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu17_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      input1_value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu18_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat5').then((res) => { 
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
    
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 384]};
    let type3_length = product(type3.dimensions);

    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);

    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);

    model.setOperandValue(axis0, new Int32Array([1]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);

    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);

    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});