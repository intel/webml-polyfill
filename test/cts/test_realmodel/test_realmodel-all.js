describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat0', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu2_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu3_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat0').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 128]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu5_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu6_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat1').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 128]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu8_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu9_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat2').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 256]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu11_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu12_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat3').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 256]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu14_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu15_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat4').then((res) => {
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu20_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu21_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat6').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 512]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for concatenation by squeezenet0_concat7', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input1_value;
    let input2_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu23_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_relu24_fwd').then((res) => {
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
    await fetch('./cts/test_realmodel/resources/squeezenet0_concat7').then((res) => {
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
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 512]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv0_fwd', async function() {
    this.timeout(60000);
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/data').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu0_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 224, 224, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 111, 111, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 3, 3, 3]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv0_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });

    let op3value;
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv0_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });

    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([2]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv10_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat2').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu10_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32, 1, 1, 256]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv10_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv10_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv11_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu10_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu11_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128, 1, 1, 32]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv11_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv11_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv12_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu10_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu12_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128, 3, 3, 32]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv12_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv12_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv13_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool2_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu13_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [48]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [48, 1, 1, 256]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv13_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv13_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv14_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu13_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu14_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192, 1, 1, 48]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv14_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv14_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv15_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu13_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu15_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192, 3, 3, 48]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv15_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv15_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv16_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat4').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu16_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [48]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [48, 1, 1, 384]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv16_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv16_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv17_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu16_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu17_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192, 1, 1, 48]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv17_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv17_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv18_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu16_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu18_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 48]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 192]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [192]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [192, 3, 3, 48]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv18_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv18_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv19_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat5').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu19_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 384]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 1, 1, 384]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv19_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv19_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv1_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool0_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu1_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [16, 1, 1, 64]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv1_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv1_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv20_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu19_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu20_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256, 1, 1, 64]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv20_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv20_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv21_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu19_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu21_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256, 3, 3, 64]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv21_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv21_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);


    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv22_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat6').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu22_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 512]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 1, 1, 512]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv22_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv22_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);


    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv23_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu22_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu23_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256, 1, 1, 64]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv23_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv23_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv24_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu22_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu24_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [256]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [256, 3, 3, 64]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv24_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv24_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv25_fwd', async function() {
    this.timeout(60000);
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_dropout0_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu25_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 512]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 1000]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1000]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1000, 1, 1, 512]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv25_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv25_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv2_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu1_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu2_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 1, 1, 16]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv2_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv2_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });

    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv3_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu1_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu3_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 3, 3, 16]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv3_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv3_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });

    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv4_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat0').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu4_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [16, 1, 1, 128]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv4_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv4_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });

    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv5_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu4_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu5_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 1, 1, 16]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv5_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv5_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv6_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu4_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu6_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 16]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [64]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [64, 3, 3, 16]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv6_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv6_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);


    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv7_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool1_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu7_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [32]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [32, 1, 1, 128]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv7_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv7_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv8_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu7_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu8_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128, 1, 1, 32]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv8_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv8_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([0]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for conv_2d by squeezenet0_conv9_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value;
    let op4_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu7_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu9_fwd').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 32]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 128]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [128]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [128, 3, 3, 32]};

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);

    let op2value;
    let op3value;

    await fetch('./cts/test_realmodel/resources/squeezenet0_conv9_weight').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    await fetch('./cts/test_realmodel/resources/squeezenet0_conv9_bias').then((res) => {
      return res.text();
    }).then((text) => {
      let arr = text.split(',');
      let file_data = new Float32Array(arr.length);
      for (let j in arr) {
        let b = parseFloat(arr[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([1]));
    model.setOperandValue(stride, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for max_pool_2d by squeezenet0_pool0_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let i0_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu0_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool0_fwd').then((res) => {
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

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 111, 111, 64]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 55, 55, 64]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for max_pool_2d by squeezenet0_pool1_fwd', async function() {
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for max_pool_2d by squeezenet0_pool2_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let i0_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_concat3').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool2_fwd').then((res) => {
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

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 27, 27, 256]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 256]};
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
describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('Check result for average_pool_2d by squeezenet0_pool3_fwd', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let i0_value;
    let output_expect;

    await fetch('./cts/test_realmodel/resources/squeezenet0_relu25_fwd').then((res) => {
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

    await fetch('./cts/test_realmodel/resources/squeezenet0_pool3_fwd').then((res) => {
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

    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1000]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 13, 13, 1000]};
    let type0_length = product(type0.dimensions);

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

    model.setOperandValue(stride, new Int32Array([5]));
    model.setOperandValue(filter, new Int32Array([100]));
    model.setOperandValue(padding, new Int32Array([50]));
    model.setOperandValue(activation, new Int32Array([0]));
    model.addOperation(nn.AVERAGE_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);

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
