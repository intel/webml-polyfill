describe('CTS Supplement Test', function() {
    const assert = chai.assert;
    const nn = navigator.ml.getNeuralNetworkContext();
  
    it('check result for CONV_2D 1 h65 w65 96 example', async function() {
      let model = await nn.createModel(options);
      let operandIndex = 0;
  
      let op1_value;
      let op2_value;
      let bias_value;
      let op3_expect;

      await fetch('./cts/test_supplement/resource/conv_65_65_96/input.txt').then((res) => {
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

      await fetch('./cts/test_supplement/resource/conv_65_65_96/filter.txt').then((res) => {
        return res.text();
      }).then((text) => {
        let arr = text.split(',');
        let file_data = new Float32Array(arr.length);
        for (let j in arr) {
          let b = parseFloat(arr[j]);
          file_data[j] = b;
        }
        op2_value = file_data;
      });

      await fetch('./cts/test_supplement/resource/conv_65_65_96/bias.txt').then((res) => {
        return res.text();
      }).then((text) => {
        let arr = text.split(',');
        let file_data = new Float32Array(arr.length);
        for (let j in arr) {
          let b = parseFloat(arr[j]);
          file_data[j] = b;
        }
        bias_value = file_data;
      });

      await fetch('./cts/test_supplement/resource/conv_65_65_96/expect.txt').then((res) => {
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

      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 65, 65, 96]};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [576, 1, 1, 96]};
      let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [576]};
      let type3 = {type: nn.INT32};
      let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 65, 65, 576]};
      let type4_length = product(type4.dimensions);
  
      let op1 = operandIndex++;
      model.addOperand(type0);
      let op2 = operandIndex++;
      model.addOperand(type1);
      let bias = operandIndex++;
      model.addOperand(type2);
      let pad = operandIndex++;
      model.addOperand(type3);
      let stride_w = operandIndex++;
      model.addOperand(type3);
      let stride_h = operandIndex++;
      model.addOperand(type3);
      let act = operandIndex++;
      model.addOperand(type3);

      let op3 = operandIndex++;
      model.addOperand(type4);
      
      model.setOperandValue(op2, new Float32Array(op2_value));
      model.setOperandValue(bias, new Float32Array(bias_value));
      model.setOperandValue(pad, new Int32Array([1]));
      model.setOperandValue(stride_w, new Int32Array([1]));
      model.setOperandValue(stride_h, new Int32Array([1]));
      model.setOperandValue(act, new Int32Array([0]));

      model.addOperation(nn.CONV_2D, [op1, op2, bias, pad, stride_w, stride_h, act], [op3]);
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
    }).timeout(50000);
  });
  