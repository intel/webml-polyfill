describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Add example', async function() {
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(float32TensorType);
    let output = operandIndex++;
    model.addOperand(float32TensorType);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    compilation.setPreference(prefer);

    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], input0Data[i] + input1Data[i]));
    }
  });

  it('check result for Concatenation axis 0 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 0;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                      109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
                      201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                      209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 1 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 1;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                      201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                      109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0,
                      209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 2 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 2;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 103.0, 104.0,
                      201.0, 202.0, 203.0, 204.0,
                      105.0, 106.0, 107.0, 108.0,
                      205.0, 206.0, 207.0, 208.0,
                      109.0, 110.0, 111.0, 112.0,
                      209.0, 210.0, 211.0, 212.0,
                      113.0, 114.0, 115.0, 116.0,
                      213.0, 214.0, 215.0, 216.0]);
    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Concatenation axis 3 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand(float32TensorType);
    let inputData1 = new Float32Array(tensorLength);
    inputData1.set([201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0,
                    209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0]);
    model.setOperandValue(1, inputData1);
    model.addOperand({type: nn.INT32});
    let axis = 3;
    model.setOperandValue(2, new Int32Array([axis]));

    let outputFloat32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 4]};
    const outputTensorLength = product(outputFloat32TensorType.dimensions);

    model.addOperand(outputFloat32TensorType);
    model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);

    model.identifyInputsAndOutputs([0], [3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                    109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(outputTensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = new Float32Array(outputTensorLength);
    expectedData.set([101.0, 102.0, 201.0, 202.0,
                      103.0, 104.0, 203.0, 204.0,
                      105.0, 106.0, 205.0, 206.0,
                      107.0, 108.0, 207.0, 208.0,
                      109.0, 110.0, 209.0, 210.0,
                      111.0, 112.0, 211.0, 212.0,
                      113.0, 114.0, 213.0, 214.0,
                      115.0, 116.0, 215.0, 216.0]);

    for (let i = 0; i < outputTensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Fully connected float 1D input example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 10, 100];
    let op3_expect = [127];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([3, 2, 1]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
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

  it('check result for Fully connected float 3D input example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 32, 16];
    let op3_expect = [8, 68, 36];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([2]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
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

  it('check result for Fully connected float 3D input example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 10, 100];
    let op3_expect = [127];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([3, 2, 1]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
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

  it('check result for Fully connected float 4D input example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 32, 16];
    let op3_expect = [8, 68, 36];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1, 1]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type4_length = product(type4.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type4);
    let act = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(op2, new Float32Array([2]));
    model.setOperandValue(b0, new Float32Array([4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
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

  it('check result for Mul example', async function() {
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;

    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(float32TensorType);
    let output = operandIndex++;
    model.addOperand(float32TensorType);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    compilation.setPreference(prefer);

    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], input0Data[i] * input1Data[i]));
    }
  });

  it('check result for Reshape example', async function() {
    let model = await nn.createModel(options);

    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions:[1, 4]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
    model.setOperandValue(1, new Int32Array([2, 2]));
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
    model.addOperation(nn.RESHAPE, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData = new Float32Array(tensorLength);
    inputData.set([1.0, 2.0, 3.0, 4.0]);
    execution.setInput(0, inputData);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], inputData[i]));
    }
  });

  it('check result for Softmax example', async function() {
    let model = await nn.createModel(options);
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.FLOAT32});
    model.setOperandValue(1, new Float32Array([1.0]));
    model.addOperand(float32TensorType);
    model.addOperation(nn.SOFTMAX, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([1.0, 1.0, 1.0, 1.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();
    let expectedData = new Float32Array(tensorLength);
    expectedData.set([0.5, 0.5, 0.5, 0.5]);

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });
});