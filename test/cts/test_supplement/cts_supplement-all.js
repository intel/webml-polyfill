describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Add example', async function() {
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    let tensorLength = product(float32TensorType.dimensions);

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

    compilation.setPreference(getPreferenceCode(options.prefer));

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

  it('check result for Add broadcasting 1D-2D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 1D-2D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-2D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-2D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 35, 36];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-2D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-2D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 21, 22];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 1D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 1D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26, 17, 28, 19, 30, 21, 32];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26, 17, 28, 19, 30, 21, 32];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 15, 16, 27, 28, 19, 20, 31, 32];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-3D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-3D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 15, 26, 37, 48, 19, 30, 41, 52];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-3D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 21, 22, 13, 14, 23, 24, 15, 16, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 15, 26, 37, 48, 19, 30, 41, 52];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 35, 46, 37, 48, 59, 70, 61, 72];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 35, 36, 47, 48, 59, 60, 71, 72];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 25, 26, 27, 28, 39, 40, 41, 42];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 15, 16, 27, 28, 19, 20, 31, 32];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/7', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26, 17, 28, 19, 30, 21, 32];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-3D example/8', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 21, 12, 22, 33, 43, 34, 44, 55, 65, 56, 66];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 1D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 1D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26, 17, 28,
                        19, 30, 21, 32, 23, 34, 25, 36];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 15, 26, 17, 28,
                        19, 30, 21, 32, 23, 34, 25, 36];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 15, 16, 27, 28,
                        19, 20, 31, 32, 23, 24, 35, 36];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 15, 26, 37, 48,
                        19, 30, 41, 52, 23, 34, 45, 56];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 2D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4,
                                       5,  6,  7,  8]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 21, 22, 13, 14, 23, 24,
                        15, 16, 25, 26, 17, 18, 27, 28];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 15, 26, 37, 48,
                        19, 30, 41, 52, 23, 34, 45, 56];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 35, 46, 37, 48,
                        19, 30, 21, 32, 43, 54, 45, 56];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 35, 36, 47, 48,
                        19, 20, 31, 32, 43, 44, 55, 56];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 55, 66, 77, 88,
                        19, 30, 41, 52, 63, 74, 85, 96];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 3D-4D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120,
                                       130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                                       250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360,
                                       370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480,
                                       490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 11,  12,  21,  22,  31,  32,  43,  44,  53,  54,  63,  64,
                         75,  76,  85,  86,  95,  96, 107, 108, 117, 118, 127, 128,
                        131, 132, 141, 142, 151, 152, 163, 164, 173, 174, 183, 184,
                        195, 196, 205, 206, 215, 216, 227, 228, 237, 238, 247, 248,
                        251, 252, 261, 262, 271, 272, 283, 284, 293, 294, 303, 304,
                        315, 316, 325, 326, 335, 336, 347, 348, 357, 358, 367, 368,
                        371, 372, 381, 382, 391, 392, 403, 404, 413, 414, 423, 424,
                        435, 436, 445, 446, 455, 456, 467, 468, 477, 478, 487, 488,
                        491, 492, 501, 502, 511, 512, 523, 524, 533, 534, 543, 544,
                        555, 556, 565, 566, 575, 576, 587, 588, 597, 598, 607, 608];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 55, 66, 77, 88,
                        19, 30, 41, 52, 63, 74, 85, 96];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 33, 44, 15, 26, 37, 48,
                        59, 70, 81, 92, 63, 74, 85, 96];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 22, 13, 24, 35, 46, 37, 48,
                        59, 70, 61, 72, 83, 94, 85, 96];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 23, 24, 35, 36, 47, 48,
                        59, 60, 71, 72, 83, 84, 95, 96];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Add broadcasting 4D-4D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.ADD, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [11, 12, 13, 14, 21, 22, 23, 24,
                        31, 32, 33, 34, 41, 42, 43, 44];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [1.85284, -0.0393656, -0.127353, 1.43115, -0.302294, -1.0402, 0.655023, -0.587614, 1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -0.346357, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.104506, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, 1.42026, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, -0.343435, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, -1.46717, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494, 1.23741, -0.527402, -0.39954, -0.0128623, 1.3644, 0.985755, -0.718118, -0.1008, 1.24327];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [-0.000614278, -1.21221, 0.443861, 0.102117, -2.52714, 1.47489, 0.173474, -0.237577, 1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, -1.6271, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.133762, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 1.22372, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, -0.838905, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 0.891823, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468, 0.166228, 0.434554, -2.57529, -0.958662, -2.23978, 2.66776, 0.542601, 1.76107, -1.08134];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.93729883, 1.2219346, 1.63162, 1.8668158, 1.6269842, -1.9670266, -0.15441051, 0.5595218, -0.99790573, 2.3631613, -1.3033884, 1.2685156, -1.054666, 0.31054103, 1.3991811, -0.46040928, -0.5490349, 0.14452362, -0.3481132, 0.62236893, -0.83281666, -3.7738001, 0.5568896, 0.9274717, 0.48187765, -0.9098393, -2.0777307, 1.213712, -0.24457066, 0.14877218, -0.5466188, 0.9753277, -0.53815746, -0.21209812, 0.43179023, 3.625693, 0.18136086, -0.61304003, 0.0709098, 1.9279834, 1.5563309, 0.9073066, 2.7159054, -2.4034908, 0.37647444, -1.606053, 1.3484854, -0.9874026, 0.13162848, -2.3492568, -2.4371247, 1.1747775, 1.2780867, -1.0992509, -0.15879333, 0.62347984, -0.39933106, 0.2999848, -1.6485932, 0.12523836, -0.4088197, 0.7373756, -0.43234983, 0.1826737];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding same example-4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [-0.19379666, 0.13141191, 0.01153314, 1.8091118, -2.3524547, 2.3089457, -0.11324078, 1.127184, 1.1204935, 2.5120242, 2.149157, 0.6553576, -0.20535097, -0.6506086, 1.8369594, 0.54811543, 0.7003816, -1.0158687, 0.26498342, 0.5115767, 0.02016348, 1.3763173, 1.6339865, -0.47847784, -0.82466245, -1.7953774, -0.58860576, -0.5705055, -1.930277, -0.10564268, 0.09756953, 0.5007975, 0.05934131, 1.4807024, -1.6880066, -0.18793303, 1.1057533, -1.0429065, 2.347888, -0.28468987, 0.63049823, -1.7537897, 3.535353, -0.15747231, 1.2116436, -0.53958964, -0.5445896, -0.05320372, 1.4167088, 0.30553037, 0.93280673, -0.04479983, 0.42253172, -1.397118, 1.622694, 2.101064, 1.3468413, -0.8429366, -0.6602505, 0.1906923, 0.3716293, 1.4599676, 1.6961491, 0.8730268];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [1.72003, 1.55816, 0.667546, 2.23663, 0.0661516, 0.290254, 0.770222, -1.58197, -0.850595, -0.484224, 0.949967, -0.577263, -0.871949, 2.34132, -0.135965, -0.985713, 0.815147, 1.03114, -1.41915, -0.515534, -0.373639, -1.50604, 0.673113, 3.06139, -0.388578, -1.76707, -0.315667, -1.03815, 0.432787, -1.41643, 1.12944, -0.175806, -0.846415, 1.40095, 0.70832, 2.19562, -2.61266, -0.705383, 1.26124, 1.46545, -2.35761, 2.04494];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [1.28735, 1.91315, 2.51734, 0.375841, 0.637563, 2.653, 2.72959, 1.17389, -2.12119, 2.91417, -2.24246, 0.0497045, -0.127107, -0.144473, -0.393284, -2.02346, -0.239178, -0.246508, 1.29277, 1.32963, 0.117521, 0.0665713, 1.09438, -1.31426, 2.52594, -0.969211, 0.515478, -1.60926, 0.135211, 0.786415, -1.14382, -0.739102, -1.01731, 0.281615, 2.36311, 1.93872, -0.150491, 3.45217, 2.28219, 1.18282, -2.25086, 3.05468];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 1]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
    model.setOperandValue(bias, new Float32Array([0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-3', async function() {
      let model = await nn.createModel(options);
      let operandIndex = 0;

      let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
      let op3_expect = [0.14452362, -0.3481132, 0.62236893, -0.83281666, -3.7738001, 0.5568896, -0.9098393, -2.0777307, 1.213712, -0.24457066, 0.14877218, -0.5466188, -0.21209812, 0.43179023, 3.625693, 0.18136086, -0.61304003, 0.0709098, 0.9073066, 2.7159054, -2.4034908, 0.37647444, -1.606053, 1.3484854];

      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
      let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
      let type3 = {type: nn.INT32};
      let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 1]};
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
      let act = operandIndex++;
      model.addOperand(type3);
      let op3 = operandIndex++;
      model.addOperand(type4);

      model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
      model.setOperandValue(bias, new Float32Array([0]));
      model.setOperandValue(pad, new Int32Array([2]));
      model.setOperandValue(rate_w, new Int32Array([2]));
      model.setOperandValue(rate_h, new Int32Array([2]));
      model.setOperandValue(act, new Int32Array([0]));

      model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

    it('check result for ATROUS_CONV_2D 1 h3 w2 implicit padding valid example-4', async function() {
      let model = await nn.createModel(options);
      let operandIndex = 0;

      let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
      let op3_expect = [-1.0158687, 0.26498342, 0.5115767, 0.02016348, 1.3763173, 1.6339865, -1.7953774, -0.58860576, -0.5705055, -1.930277, -0.10564268, 0.09756953, 1.4807024, -1.6880066, -0.18793303, 1.1057533, -1.0429065, 2.347888, -1.7537897, 3.535353, -0.15747231, 1.2116436, -0.53958964, -0.5445896];

      let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
      let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 2, 3]};
      let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
      let type3 = {type: nn.INT32};
      let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 1]};
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
      let act = operandIndex++;
      model.addOperand(type3);
      let op3 = operandIndex++;
      model.addOperand(type4);

      model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203, -0.579455, 0.0278809, -0.79946, -0.684259, 0.563238, 0.37289, 0.738216, 0.386045, -0.917775, 0.184325, -0.270568, 0.82236, 0.0973683, -0.941308, -0.144706]));
      model.setOperandValue(bias, new Float32Array([0]));
      model.setOperandValue(pad, new Int32Array([2]));
      model.setOperandValue(rate_w, new Int32Array([2]));
      model.setOperandValue(rate_h, new Int32Array([2]));
      model.setOperandValue(act, new Int32Array([0]));

      model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.27853, 1.74987, -0.876718, 0.989692, 0.298548, 0.522103, -0.536896, -0.179382, -0.966914, 1.33708, 1.37042, -0.495494, 1.43859, -1.548, -0.430026, -0.662793, -0.0867897, -0.900658, -0.524396, 0.255731, -0.779081, 0.12666, 0.915651, -0.444765, -0.186842, -1.87308, 1.21135, -0.385009, 1.72032, -1.56036, -1.23059, 1.23694, 0.00200015, 0.359522, 1.60084, 0.434006, -0.282945, 2.37292, -1.28653, 0.0847837, -0.352093, -2.39659, 0.149246, 0.920351, -1.34346, 0.952311, -0.35811, 0.403449, 0.484796, -1.19989, -0.684298, -1.41301, 0.103177, -0.307039, 1.17741, 2.58936, -2.76237, -1.21565, -1.09619, 1.17432, 0.512143, 0.771379, 0.399879, -0.0533093, 0.290864, 0.95563, 1.16328, 1.80768, -1.52564, -0.126476, -0.185224, -0.114779, 1.2248, 0.237127, -0.213297, -0.619941, 0.497944, -1.68688, 1.59314, -0.127337, 0.111419, 1.13719, 1.68537, -0.479644, 1.18608, -2.52744, 1.34136, 0.548297, -2.0838, 2.64585, -0.993354, 0.128238, 1.26092, 0.318668, 0.893795, -0.0600559, -0.629126, -0.949229, 2.25828, -1.961, 0.00589599, -0.187854, -1.02403, 0.396121, 1.3704, 3.99355, 0.434221, 0.274464, -0.562438, -0.914871, 0.539129, -0.928687, 0.834954, 0.844178, -0.566053, -0.957341, 0.933336, 1.13613, -1.22109, 1.4649, -0.414666, -0.452821, -0.706006, -1.72657, -0.726574, -0.0979362, -0.478669, 1.78703, -0.639288, 1.48565, -0.179904, 1.01003, -0.317118, -0.675387, 1.90969, -1.38343, 0.697255, -0.292255, 1.81634, 0.717801, 0.862479, -0.407478, -0.343106, -0.0353232, -0.481893, -0.135565, -2.95941, 0.247846, 2.67757, -2.23999, -0.519673, 0.254447, 0.415283, -1.01065, 0.507911, 0.979926, -0.184304, -0.000950437, -0.734348, -0.196685, -0.713241, 0.594972, 0.0845042, 2.48496, 0.385019, -0.201145, 0.533332, -0.904872, -0.333518, -0.581063, -2.07065, 0.118687, -1.86708, -0.601987, 0.432037, 1.73923, 0.590007, 0.419788, 0.314198, 2.12817, 0.570793, -1.15998, -0.348587, -1.10231, -2.13091, 0.134467, -0.460382, 0.138338, 3.455, 0.679068, -0.190282, -0.0307461];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [0.78574, 0.0700466, -0.110245, 0.0141003, -0.621007, -0.979104, 1.24104, 0.580398, -0.512997, 0.900559, -0.683229, -1.0162, 1.0089, -0.0752488, 0.110969, 0.270558, 0.756819, -0.10753, -0.371484, 0.149005, 0.0973829, 0.155766, -0.476502, 0.259481, 1.06709, -1.16534, 1.52694, -0.797245, 0.802736, -0.997109, 2.2661, -1.45548, 2.15506, -1.33682, 1.15225, -3.09324, 0.943457, 0.885211, 0.987944, -0.345875, -0.114708, 1.7107, 0.104745, 0.828324, -2.49964, -0.453742, -0.288829, -0.0948694, -0.489415, 1.74889, -0.378257, -2.10237, 0.613022, -2.5225, -0.746785, 3.63816, -1.9287, 0.774279, -0.613917, -0.650011, 1.03753, -0.177923, 0.891815, -1.00373, 1.83859, -1.59239, -0.0662623, 0.218806, -1.088, 0.280837, 0.902901, -1.90127, 3.04734, -1.57302, 1.10881, -0.980369, -3.85305, -0.955859, 1.64909, 2.33573, 0.31144, -0.594375, 0.325747, -0.952566, -0.613449, 2.85073, 1.94692, 1.12977, 1.1351, -0.449652, 0.118765, -0.199547, 2.873, 1.35182, -1.85457, 1.22364, 1.38049, 2.38342, 0.882321, 1.03795, -0.321571, -2.60202, -1.6372, 1.09302, 0.461768, 1.8485, -0.158928, 4.28871, -0.437375, -1.5794, 1.59869, 0.0811864, 0.912054, 0.452176, 2.01812, 2.62907, 1.50304, -0.840276, -0.455854, -0.224913, 0.609824, -0.11105, 3.35635, 2.02386, 1.4687, -0.708365, -0.508992, -3.02602, -0.75725, 1.85277, 2.92817, -0.172997, -1.13279, -0.355636, -0.337669, -0.588752, 2.05759, 1.0651, 0.884758, -0.0712112, 3.81319, 0.771629, 0.949634, 0.0838967, -2.19264, 0.114521, 0.543556, -1.63197, -0.267442, 1.15701, -2.37862, 2.57646, 0.531208, 0.9499, -0.231441, 1.51461, 1.58888, 0.895931, -0.753084, 0.545251, 0.746903, 0.012994, -0.790398, -1.1055, 1.77789, 0.430923, 0.818241, -0.731412, 0.979546, -2.48707, -1.53658, -1.66798, -1.04585, -0.667911, 1.00299, -2.20339, 0.137826, -2.31281, 0.755535, 0.495396, 0.549629, 0.713128, 0.751369, 0.283996, -0.814532, 1.4866, 1.12105, 0.927998, 0.517938, -0.612661, -1.47756, -1.42422];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.1709702, -0.15783468, 0.23686486, 0.18489599, 2.4330406, -0.83415174, -0.9379462, 2.3037708, -1.8530072, -0.6243031, -0.5570223, 0.0571152, -0.08371013, 1.2763771, -1.0637468, -0.034401, 0.2984934, -0.37836313, 0.0089688, 0.317119, -1.6136328, -0.04100603, -0.14438102, 0.42039546, 1.0826977, -1.4740779, 0.93527263, -1.6937343, 0.9933808, -0.00613139, 1.122337, -1.0007913, -0.19323519, -1.536109, 1.7444146, 0.07549044, 0.14941335, -1.0424318, 1.6394564, 0.05120939, 0.44861174, -1.0254043, 0.2891852, 1.0087392, -1.6427121, 0.22787222, 0.7912595, -1.0432496, 0.3554331, -1.2678999, 0.8390462, 2.43199, 0.7855995, -0.89247733, -0.8421999, -0.21892299, -1.4752842, 1.4666033, 1.2671402, -1.9113951, 2.8163433, 0.42375302, 1.384544, -0.05179526, -0.09704095, -1.0454197, -0.7849326, -1.0726444, -2.2269573, 0.38411486, -0.15826067, 1.7655121, -0.21607418, -0.220653, -1.0505613, -1.9059162, -1.3809854, -0.21753564, -0.6674532, 0.9924352, -1.3004371, 1.3581562, -0.50957847, 0.43931735, -0.30051446, 1.9288344, 1.3749437, 0.24674952, -1.3658104, -0.24712396, 1.8478253, 0.0548588, 0.5765619, 0.12883057, -0.9403651, 0.8154919, -0.38991603, 0.2580973, 0.27158368, -1.0782311, 2.7078195, 0.54151404, -1.2969424, -0.4957502, -0.8728107, 2.7895741, 0.764437, 1.8849254, -0.16873728, 0.36533558, 2.3231673, -1.0529735, -1.2732302, 0.87934554, 1.3826215, 0.24184477, 1.3531275, -0.28793597, 2.0084376, 1.4573742, -1.5291485, 0.31902915, -0.23054239, 0.62534, 1.8519323, -2.245485, -1.8446102, 0.66178447, -1.6817732, 0.43443537, -1.101484, 0.8291666, 0.7223018, -0.18338689, 1.9866216, -0.7683655, -1.1324087, -0.671756, -0.99642277, 1.714391, -0.30889648, -1.1144117, 0.58786345, -1.4462819, 0.5452746, -1.4152023, -0.51632243, -1.0784085, -1.8019311, 1.8430812, -0.77855986, 1.8445983, -1.4430277, 0.40093422, 1.7084532, 0.8918805, 0.36253592, -0.4176629, 0.91448, -0.92981076, -0.07481962, 0.8215766, 0.31338146, 0.26393196, -1.0675564, 0.70066214, 0.31446722, 0.87955433, 0.4141644, -0.8118956, -1.1245772, 1.742084, 0.3557291, -2.3003993, -0.01551861, -0.8920257, -0.8597669, -0.8725518, 0.6303311, 1.0367441, -0.6833122, -1.3960947, -0.16622084, 0.06926525, -1.2923226, 0.53113765, -0.04628024, 0.63293314, 2.2518039, -0.3721881, -1.2018218, 0.750074];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding same example-4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [5.68123579e-01, -1.14592409e+00, 3.55027258e-01, 2.65643388e-01, 2.86270237e+00, -2.05537200e+00, 9.96777594e-01, -1.54748714e+00, 6.60102963e-02, -7.71302223e-01, 2.64946890e+00, -1.73251033e+00, 2.21911097e+00, -2.69358099e-01, 4.78585958e-02, -2.20272660e+00, 2.34142375e+00, -1.37358618e+00, 8.28311384e-01, 1.88295513e-01, 4.20671701e-01, -8.09967041e-01, 1.62417030e+00, -5.29218435e-01, -2.15015352e-01, 6.98525012e-01, 6.55916929e-02, -2.56302881e+00, -1.71343148e-01, -3.45137477e-01, -1.05851889e-01, 1.45706737e+00, -4.94342566e-01, 1.08996737e+00, 3.88428628e-01, -2.61325389e-02, -9.89482582e-01, 4.57503766e-01, 9.77390289e-01, -1.50680214e-01, -7.15103522e-02, -1.05187809e+00, -4.44289744e-01, -1.06116939e+00, 2.08077133e-01, -4.67302501e-01, 2.66760141e-01, -2.81759053e-02, 1.89035749e+00, -2.55258560e-01, 7.55438805e-02, -1.15671694e-01, -6.44194603e-01, 3.80814433e-01, 8.10133100e-01, -2.17651606e-01, -1.79334271e+00, 2.35288119e+00, 3.08204472e-01, 1.39144015e+00, -2.16177559e+00, 2.01937342e+00, -1.95060742e+00, 1.08799064e+00, 1.07404935e+00, -8.81704211e-01, 2.31657219e+00, 1.85450721e+00, 3.53201330e-01, 1.30680501e-01, -1.78767276e+00, -1.31494522e-01, 1.81664824e+00, 2.56881714e-02, 6.11234307e-01, 3.47290456e-01, -4.14338410e-01, -4.65534687e-01, 9.53575492e-01, 1.16449952e+00, -1.43737674e+00, -8.24268520e-01, 1.25888991e+00, -1.35992676e-01, -3.36073041e-01, -1.68412328e-01, 3.38715374e-01, 1.26111448e-01, 3.44100773e-01, 1.78472698e-01, 1.32074022e+00, 9.62635517e-01, 7.05493331e-01, -3.74882787e-01, -2.35561743e-01, 4.73342448e-01, 7.13551819e-01, 3.78164232e-01, 8.37010503e-01, 7.96169400e-01, -9.82071996e-01, -7.78258145e-01, -3.81720662e-01, 4.97114360e-02, -9.41047251e-01, 1.28766775e+00, -2.09531784e-01, 1.22835684e+00, -7.96112239e-01, 9.58650827e-01, 4.83479738e-01, 5.86519122e-01, -7.83452988e-02, -7.24613249e-01, 8.10349941e-01, 2.37005234e-01, 2.26917839e+00, -2.95450389e-01, 1.76701534e+00, 4.67392266e-01, -1.56397986e+00, 1.32009792e+00, 6.34528875e-01, -1.22586346e+00, 8.30167711e-01, -9.60741878e-01, -1.65419698e+00, 1.88168168e+00, -7.61278868e-01, -1.00265503e-01, 1.68877339e+00, 2.23001170e+00, 1.04707837e+00, 1.34341955e+00, 8.66675556e-01, 1.07389760e+00, -5.54591119e-01, -1.18077368e-01, -2.32217640e-01, 7.40962386e-01, 1.63370538e+00, 1.70431405e-01, 5.72544217e-01, 1.13617229e+00, -5.35305738e-01, 1.39170408e-01, 1.78036332e+00, 1.40088332e+00, 1.83663023e+00, -5.23582458e-01, -5.13517082e-01, -2.47991228e+00, 8.69209886e-01, 1.78403348e-01, -3.98485154e-01, -9.86580372e-01, -4.61422592e-01, -1.53689396e+00, 1.18083906e+00, -1.82503152e+00, 1.72589660e+00, -7.58466005e-01, 4.05678511e-01, 1.15027118e+00, 1.01947999e+00, 1.86936712e+00, 6.04811609e-02, 1.61855984e+00, -5.02297580e-02, -1.50424528e+00, 1.73169747e-03, -8.82876217e-01, -2.38876629e+00, 5.89107275e-01, 4.73287106e-02, 7.63606846e-01, -1.45487642e+00, -7.85828531e-01, 1.31718516e-02, 3.03435946e+00, 1.15426493e+00, -7.38213897e-01, 4.62800056e-01, 4.93960440e-01, 3.14571261e-01, 1.90042281e+00, -3.42714101e-01, -1.07730091e+00, 1.19726253e+00, 2.30893791e-01, 2.67316192e-01, 1.05705726e+00];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([1]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [-1.86841726e-01, -1.87308407e+00, 1.21135116e+00, -3.85009050e-01, 1.72032380e+00, -1.56035602e+00, -1.23059344e+00, 1.23694098e+00, 1.99985504e-03, 3.59522343e-01, 1.60084629e+00, 4.34007555e-01, -2.82945693e-01, 2.37292123e+00, -1.28653407e+00, 8.47842395e-02, -3.52093250e-01, -2.39659071e+00, 1.49246454e-01, 9.20351386e-01, -1.34345913e+00, 4.84796733e-01, -1.19989347e+00, -6.84298515e-01, -1.41301155e+00, 1.03178442e-01, -3.07042211e-01, 1.17741525e+00, 2.58936214e+00, -2.76237011e+00, -1.21565342e+00, -1.09619403e+00, 1.17431641e+00, 5.12142301e-01, 7.71379948e-01, 3.99879634e-01, -5.33092022e-02, 2.90863872e-01, 9.55634058e-01, 1.16327548e+00, 1.80768192e+00, -1.52564144e+00, 1.22480464e+00, 2.37127364e-01, -2.13295698e-01, -6.19941294e-01, 4.97942507e-01, -1.68688416e+00, 1.59314167e+00, -1.27335250e-01, 1.11420155e-01, 1.13719368e+00, 1.68536687e+00, -4.79643047e-01, 1.18607867e+00, -2.52744436e+00, 1.34135664e+00, 5.48298419e-01, -2.08380222e+00, 2.64585400e+00, -9.93354917e-01, 1.28238201e-01, 1.26091874e+00, -6.29126132e-01, -9.49230671e-01, 2.25827789e+00, -1.96100128e+00, 5.89534640e-03, -1.87852085e-01, -1.02403378e+00, 3.96120340e-01, 1.37040257e+00, 3.99355221e+00, 4.34221208e-01, 2.74464667e-01, -5.62437356e-01, -9.14871454e-01, 5.39128900e-01, -9.28685188e-01, 8.34952950e-01, 8.44179749e-01, -5.66052437e-01, -9.57342565e-01, 9.33336258e-01, -4.14666116e-01, -4.52821493e-01, -7.06006944e-01, -1.72656703e+00, -7.26575494e-01, -9.79378521e-02, -4.78667945e-01, 1.78702688e+00, -6.39287651e-01, 1.48564780e+00, -1.79904699e-01, 1.01003110e+00, -3.17118764e-01, -6.75386369e-01, 1.90969336e+00, -1.38342953e+00, 6.97255731e-01, -2.92255253e-01, 1.81634486e+00, 7.17801273e-01, 8.62478435e-01, -4.81892645e-01, -1.35565460e-01, -2.95940900e+00, 2.47845054e-01, 2.67756557e+00, -2.23998690e+00, -5.19674301e-01, 2.54447937e-01, 4.15283501e-01, -1.01065040e+00, 5.07912159e-01, 9.79926169e-01, -1.84304118e-01, -9.52005386e-04, -7.34347284e-01, -1.96684420e-01, -7.13242233e-01, 5.94973564e-01, 8.45057964e-02, 2.48496294e+00, 3.85019749e-01];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [1.0670944, -1.1653414, 1.5269374, -0.797246, 0.8027355, -0.9971091, 2.2661014, -1.4554825, 2.1550565, -1.3368183, 1.152245, -3.0932455, 0.9434587, 0.88521, 0.98794407, -0.34587651, -0.11470786, 1.7107, 0.10474667, 0.8283235, -2.4996405, -0.4894141, 1.7488927, -0.3782575, -2.102374, 0.6130228, -2.5224953, -0.74678445, 3.6381645, -1.9287052, 0.77427876, -0.6139177, -0.6500135, 1.0375304, -0.1779207, 0.8918158, -1.003732, 1.8385941, -1.5923887, -0.06626323, 0.21880639, -1.0879987, 3.04734, -1.5730183, 1.1088114, -0.98036975, -3.8530445, -0.95585847, 1.649088, 2.335727, 0.31143993, -0.5943746, 0.32574737, -0.9525648, -0.61344874, 2.8507283, 1.9469215, 1.1297737, 1.1351031, -0.44965148, 0.11876422, -0.19954622, 2.872999, 1.3804917, 2.383419, 0.88232017, 1.0379514, -0.321571, -2.602018, -1.6372006, 1.0930251, 0.4617672, 1.8484964, -0.15892673, 4.2887144, -0.4373755, -1.579401, 1.5986867, 0.08118564, 0.91205466, 0.45217666, 2.018124, 2.6290717, 1.5030414, 0.60982406, -0.11105055, 3.3563545, 2.0238643, 1.468704, -0.7083628, -0.5089922, -3.0260181, -0.7572474, 1.8527693, 2.9281664, -0.17299666, -1.132788, -0.35563594, -0.33767, -0.58875316, 2.0575933, 1.0651011, 0.8847578, -0.07121107, 3.8131914, -2.1926446, 0.11451936, 0.543556, -1.6319685, -0.2674431, 1.1570106, -2.3786144, 2.5764546, 0.53120923, 0.9498998, -0.23144162, 1.5146098, 1.5888821, 0.89593095, -0.7530838, 0.54525167, 0.74690276, 0.01299442, -0.79039794, -1.1055036, 1.7778882];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 6, 7, 3]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [2.43199, 0.7855995, -0.89247733, -0.8421999, -0.21892299, -1.4752842, 1.4666033, 1.2671402, -1.9113951, 2.8163433, 0.42375302, 1.384544, -0.05179526, -0.09704095, -1.0454197, -0.7849326, -1.0726444, -2.2269573, -1.9059162, -1.3809854, -0.21753564, -0.6674532, 0.9924352, -1.3004371, 1.3581562, -0.50957847, 0.43931735, -0.30051446, 1.9288344, 1.3749437, 0.24674952, -1.3658104, -0.24712396, 1.8478253, 0.0548588, 0.5765619, -1.0782311, 2.7078195, 0.54151404, -1.2969424, -0.4957502, -0.8728107, 2.7895741, 0.764437, 1.8849254, -0.16873728, 0.36533558, 2.3231673, -1.0529735, -1.2732302, 0.87934554, 1.3826215, 0.24184477, 1.3531275, 0.62534, 1.8519323, -2.245485, -1.8446102, 0.66178447, -1.6817732, 0.43443537, -1.101484, 0.8291666, 0.7223018, -0.18338689, 1.9866216, -0.7683655, -1.1324087, -0.671756, -0.99642277, 1.714391, -0.30889648];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 3]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_CONV_2D 3 h3 w2 implicit padding valid example-4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [-0.11567169, -0.6441946, 0.38081443, 0.8101331, -0.2176516, -1.7933427, 2.3528812, 0.30820447, 1.3914402, -2.1617756, 2.0193734, -1.9506074, 1.0879906, 1.0740494, -0.8817042, 2.3165722, 1.8545072, 0.35320133, 0.34729046, -0.4143384, -0.4655347, 0.9535755, 1.1644995, -1.4373767, -0.8242685, 1.2588899, -0.13599268, -0.33607304, -0.16841233, 0.33871537, 0.12611145, 0.34410077, 0.1784727, 1.3207402, 0.9626355, 0.70549333, 0.7961694, -0.982072, -0.77825814, -0.38172066, 0.04971144, -0.94104725, 1.2876678, -0.20953178, 1.2283568, -0.79611224, 0.9586508, 0.48347974, 0.5865191, -0.0783453, -0.72461325, 0.81034994, 0.23700523, 2.2691784, -1.2258635, 0.8301677, -0.9607419, -1.654197, 1.8816817, -0.76127887, -0.1002655, 1.6887734, 2.2300117, 1.0470784, 1.3434196, 0.86667556, 1.0738976, -0.5545911, -0.11807737, -0.23221764, 0.7409624, 1.6337054];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3, 2, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4, 6, 3]};
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
    let act = operandIndex++;
    model.addOperand(type3);
    let op3 = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.579455, -0.684259, 0.738216, 0.184325, 0.0973683, -0.176863, -0.23936, -0.000233404, 0.055546, -0.232658, -0.316404, -0.012904, 0.320705, -0.326657, -0.919674, 0.868081, -0.824608, -0.467474, 0.0278809, 0.563238, 0.386045, -0.270568, -0.941308, -0.779227, -0.261492, -0.774804, -0.79665, 0.22473, -0.414312, 0.685897, -0.327792, 0.77395, -0.714578, -0.972365, 0.0696099, -0.82203, -0.79946, 0.37289, -0.917775, 0.82236, -0.144706, -0.167188, 0.268062, 0.702641, -0.412223, 0.755759, 0.721547, -0.43637, -0.274905, -0.269165, 0.16102, 0.819857, -0.312008]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(act, new Int32Array([0]));

    model.addOperation(nn.ATROUS_CONV_2D, [op1, op2, bias, pad, rate_w, rate_h, act], [op3]);
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

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.840539, -0.301347, 0.754947, -0.14848, -0.40603, 0.294432, 0.130372, 0.11573, -0.182277, 0.2504, 0.132901, 0.442306, -0.739693, -0.196274, 0.457246, -0.636246, -0.100205, 0.698864, 0.244348, 0.22389, -0.436108, 0.067699, 0.462205, 0.249125, -0.145748, -0.387964, -0.391573, -0.392801, 0.166114, -0.622396, 0.344322, -0.374205, 0.586815, -0.203372, 0.29652, -0.590411, 0.726629, -0.213891, 0.452749, 0.532555, -0.208851, 0.186981, -0.209039, 0.398664, 0.288932, -0.540171, 0.312503, 0.24948, 0.351473, 0.076122, -0.0576253, -0.73055, 0.0665069, -0.271043, 0.634142, 0.466579, 0.536743, 0.389538, 0.417773, -0.355728, -0.591672, 0.40651, 0.586329, 0.384641, 0.0198003, -0.358878, 0.894009, -0.0825318, -0.676451, -0.0935613, 0.138747, 0.351167, -0.0305845, 0.118962, -0.201319, -0.0916215, -0.300945, 0.743041, -0.34075, 0.421278, -0.218791, 0.935359, 0.287684, 0.319749, -0.907324, 0.054362, -0.0883874, 0.0563023, -0.203432, -0.275113, -0.444178, -0.335382, -0.408242, 0.657194, 0.194033, -0.279365, -0.488907, 0.157917, 0.0881365, 0.166668, -0.407001, -0.766027, 0.921655, -0.422149, -0.624807, -0.202641, 0.13341, 0.374139, -0.109369, -0.0353696, -0.0759913, 0.456887, -0.44906, 0.131841, 0.811082, -0.213681, -0.134277, -0.333215, 0.0615286, -0.566144, 0.522554, -0.267049, 0.785754, -0.489062, 0.0728509, -0.0649092, -0.731203, 0.3095, -0.199677, -0.445251, -0.0831503, 0.238257, 0.618959, -0.328446, 0.416281, 0.549062, 0.0333644, -0.340149, -0.154278, 0.142677, -0.110001, 0.15484, -0.368053, 0.619189, -0.580424, -0.123033, 0.133487, -0.461813, 0.328611, 0.600933, 0.907739, 0.245199, -0.767835, 0.49435, 0.235373, -0.0873295, 0.312748, -0.249839, 0.693584, -0.351866, -0.0173133, 0.13876, 0.39089, 0.380607, -0.754171, 0.322982, -0.312857, 0.658611, -0.151223, 0.200055, -0.311675, -0.790939, 0.303812, -0.351079, 0.566216, 0.261687, 0.68551, -0.0862257, 0.290419, -0.175771, -0.449781, -0.2199, -0.312586, -0.399111, -0.0845297, -0.142101, -0.575998, -0.385935, -0.544937, 0.680582, 0.139135, -0.573594];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([1]));
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

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [0.285357, 0.00181194, 0.453967, -0.160473, 0.133146, 0.125066, 0.695562, 0.406415, 0.612903, -0.796108, -0.221201, 0.272369, -0.181291, -0.0199411, 0.679734, 0.729573, 0.22086, 0.0192072, -0.0467102, -0.436349, 0.790771, -0.0121533, -0.102724, -0.281631, 0.146536, -0.0437044, -0.643831, -0.125283, -0.392138, 0.223089, -0.893282, -0.16027, -0.22558, -0.338964, -0.393444, 0.447179, 0.0027382, 0.0600548, 0.5614, 0.308335, -0.395642, -0.232637, -0.317546, -0.0137323, 0.0275952, -0.571289, 0.0347555, 0.609347, -0.446445, 0.27283, 0.485148, -0.602337, -0.250224, 0.551432, 0.923353, 0.360036, -0.394563, -0.64193, -0.18673, 0.796443, 0.266929, 0.421638, -0.44727, 0.926579, -0.22563, 0.663612, -0.295051, 0.44308, -0.680228, 0.36995, 0.376663, 0.654893, 0.289675, 0.107439, -0.673064, 0.0995729, 0.213019, 0.18728, -0.525372, 0.449116, -0.778254, 0.82822, 0.450766, 0.24037, 0.691436, -0.357748, 0.3905, 0.570203, 0.111496, -0.553228, 0.457363, 0.149417, -0.769431, -0.470166, -0.271529, -0.349652, 0.773931, -0.135924, 0.406866, 0.426256, -0.335963, 0.680992, -0.936889, -3.52306e-05, 0.575398, 0.509084, 0.16487, -0.657185, -0.321545, -0.338165, -0.335108, 0.902524, 0.133092, -0.790369, 0.676731, 0.46084, 0.489389, 0.66835, -0.231156, 0.0692682, -0.0377757, -0.139746, 0.105297, 0.938753, -0.403865, -0.222446, 0.45314, 0.119956, -0.388121, 0.26389, 0.27597, 0.679432, 0.700873, 0.0910737, 0.213449, 0.0917136, 0.0842865, -0.0367311, 0.214628, 0.188827, -0.243133, 0.898085, -0.271172, 0.139627, -0.319151, -0.00811307, 0.522665, -0.459861, -0.424081, -0.19957, 0.494902, -0.169442, -0.0407964, -0.629691, -0.462826, -0.567803, 0.453167, 0.0473601, 0.562038, 0.152508, 0.316812, 0.582181, 0.637157, 0.190546, -0.556541, -0.860239, -0.106728, 0.616123, -0.746842, -0.0255713, -0.453518, -0.886067, 0.418399, 0.577391, -0.467784, -0.05079, -0.685036, -0.462692, 0.460047, -0.318271, 0.708224, -0.351821, -0.364416, 0.0954479, -0.0586282, -0.0894044, 0.481278, -0.201991, -0.283279, -0.897555, 0.0611137, 0.0467872];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([1]));
    model.setOperandValue(rate_h, new Int32Array([1]));
    model.setOperandValue(mul, new Int32Array([1]));
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

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.869931, 0.644628, -0.918393, 0.153672, 0.868562, -0.358177, -0.134931, -0.247565, 0.22174, -0.259157, -0.284296, -0.538065, 0.765559, 0.41986, -0.556241, 0.658494, 0.214355, -0.850169, -0.252893, -0.478935, 0.530526, -0.0700663, -0.988729, -0.303061, 0.150845, 0.829915, 0.476349, 0.406537, -0.355343, 0.757145, -0.356362, 0.800482, -0.713861, 0.210483, -0.634303, 0.718236, -0.752038, 0.457547, -0.550769, -0.551178, 0.446766, -0.227462, 0.216348, -0.852806, -0.351486, 0.55906, -0.668493, -0.303493, -0.363763, -0.162837, 0.0701012, 0.756097, -0.142269, 0.329724, -0.656317, -0.998086, -0.652949, -0.40316, -0.893682, 0.432744, 0.612362, -0.869588, -0.71327, -0.398092, -0.0423559, 0.436576, -0.925272, 0.176549, 0.822904, 0.096833, -0.296802, -0.427195, 0.031654, -0.254479, 0.244905, 0.0948254, 0.643769, -0.90391, 0.352665, -0.901179, 0.266159, -0.968068, -0.615401, -0.388975, 0.939052, -0.116289, 0.107523, -0.0582711, 0.435172, 0.334675, 0.459711, 0.717436, 0.496627, -0.680175, -0.415066, 0.339848, 0.506004, -0.337808, -0.107218, -0.172496, 0.870638, 0.931872, -0.953884, 0.903042, 0.760078, 0.209727, -0.285384, -0.45514, 0.113194, 0.0756611, 0.0924435, -0.472863, 0.960609, -0.160385, -0.839445, 0.457097, 0.163348, 0.344867, -0.131619, 0.688715, -0.540827, 0.571259, -0.95587, 0.506164, -0.155839, 0.0789621, 0.756772, -0.662069, 0.242908, 0.460821, 0.177872, -0.289839, -0.640603, 0.702598, -0.506406, -0.568262, -0.0713716, 0.413792, 0.159673, -0.305208, 0.133816, -0.160254, 0.787323, -0.753244, 0.600721, 0.263186, -0.162387, 0.477962, -0.702951, -0.731036, -0.939481, -0.524519, 0.934072, -0.511637, -0.503499, 0.106236, -0.323684, 0.534444, -0.843745, 0.364171, 0.0370358, -0.168801, -0.404559, -0.814178, 0.91745, -0.334276, 0.66925, -0.801201, 0.156511, -0.427949, 0.379153, 0.818597, -0.649902, 0.427087, -0.586015, -0.559789, -0.833923, 0.0892409, -0.621251, 0.213826, 0.465509, 0.4704, 0.380261, 0.413067, 0.180822, 0.172866, 0.59614, 0.825575, 0.662916, -0.704381, -0.297631, 0.697778];
    let op3_expect = [0.840539, -0.301347, 0.754947, -0.14848, -0.40603, 0.294432, 0.130372, 0.11573, -0.182277, 0.2504, 0.132901, 0.442306, -0.739693, -0.196274, 0.457246, -0.636246, -0.100205, 0.698864, 0.244348, 0.22389, -0.436108, 0.067699, 0.462205, 0.249125, -0.145748, -0.387964, -0.391573, -0.392801, 0.166114, -0.622396, 0.344322, -0.374205, 0.586815, -0.203372, 0.29652, -0.590411, 0.726629, -0.213891, 0.452749, 0.532555, -0.208851, 0.186981, -0.209039, 0.398664, 0.288932, -0.540171, 0.312503, 0.24948, 0.351473, 0.076122, -0.0576253, -0.73055, 0.0665069, -0.271043, 0.634142, 0.466579, 0.536743, 0.389538, 0.417773, -0.355728, -0.591672, 0.40651, 0.586329, 0.384641, 0.0198003, -0.358878, 0.894009, -0.0825318, -0.676451, -0.0935613, 0.138747, 0.351167, -0.0305845, 0.118962, -0.201319, -0.0916215, -0.300945, 0.743041, -0.34075, 0.421278, -0.218791, 0.935359, 0.287684, 0.319749, -0.907324, 0.054362, -0.0883874, 0.0563023, -0.203432, -0.275113, -0.444178, -0.335382, -0.408242, 0.657194, 0.194033, -0.279365, -0.488907, 0.157917, 0.0881365, 0.166668, -0.407001, -0.766027, 0.921655, -0.422149, -0.624807, -0.202641, 0.13341, 0.374139, -0.109369, -0.0353696, -0.0759913, 0.456887, -0.44906, 0.131841, 0.811082, -0.213681, -0.134277, -0.333215, 0.0615286, -0.566144, 0.522554, -0.267049, 0.785754, -0.489062, 0.0728509, -0.0649092, -0.731203, 0.3095, -0.199677, -0.445251, -0.0831503, 0.238257, 0.618959, -0.328446, 0.416281, 0.549062, 0.0333644, -0.340149, -0.154278, 0.142677, -0.110001, 0.15484, -0.368053, 0.619189, -0.580424, -0.123033, 0.133487, -0.461813, 0.328611, 0.600933, 0.907739, 0.245199, -0.767835, 0.49435, 0.235373, -0.0873295, 0.312748, -0.249839, 0.693584, -0.351866, -0.0173133, 0.13876, 0.39089, 0.380607, -0.754171, 0.322982, -0.312857, 0.658611, -0.151223, 0.200055, -0.311675, -0.790939, 0.303812, -0.351079, 0.566216, 0.261687, 0.68551, -0.0862257, 0.290419, -0.175771, -0.449781, -0.2199, -0.312586, -0.399111, -0.0845297, -0.142101, -0.575998, -0.385935, -0.544937, 0.680582, 0.139135, -0.573594];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([1]));
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

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D example-4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [-0.295335, -0.00387601, -0.552251, 0.166084, -0.28482, -0.152143, -0.719885, -0.869386, -0.745598, 0.823947, 0.473183, -0.331337, 0.187631, 0.0426571, -0.826897, -0.755085, -0.472453, -0.0233656, 0.0483436, 0.933418, -0.961974, 0.0125783, 0.219742, 0.342604, -0.15166, 0.0934905, 0.783221, 0.129664, 0.838844, -0.271388, 0.924519, 0.342843, 0.274418, 0.350817, 0.841638, -0.543993, -0.00283395, -0.128467, -0.682943, -0.319117, 0.84634, 0.283003, 0.32865, 0.0293755, -0.0335696, 0.591266, -0.0743476, -0.741271, 0.462056, -0.583625, -0.590183, 0.6234, 0.535269, -0.670818, -0.955642, -0.770173, 0.479986, 0.664377, 0.399445, -0.968874, -0.276263, -0.901951, 0.544104, -0.958981, 0.482658, -0.807284, 0.305369, -0.947818, 0.827498, -0.382887, -0.805741, -0.796678, -0.299804, -0.229828, 0.818783, -0.103055, -0.45568, -0.227827, 0.543743, -0.96073, 0.946747, -0.857182, -0.96426, -0.292411, -0.715614, 0.765278, -0.475043, -0.590142, -0.238507, 0.673002, -0.473357, -0.319626, 0.936014, 0.486607, 0.580844, 0.425352, -0.800994, 0.290763, -0.494953, -0.441162, 0.718677, -0.828427, 0.96965, 7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466, 0.332789, 0.723389, 0.407659, -0.934084, -0.284705, 0.961484, -0.700395, -0.985808, -0.595342, -0.691721, 0.49448, -0.0842649, 0.0390966, 0.298938, -0.128094, -0.97158, 0.86393, 0.270606, -0.468986, -0.256605, 0.47215, -0.273117, -0.590343, -0.826529, -0.725381, -0.194821, -0.259661, -0.0949207, -0.180302, 0.0446834, -0.222133, -0.40393, 0.295772, -0.92949, 0.580079, -0.169856, 0.330311, 0.0173551, -0.635823, 0.475942, 0.907175, 0.242777, -0.512208, 0.362463, 0.0496289, 0.65171, 0.990057, 0.690733, -0.469013, -0.101311, -0.68372, -0.157841, -0.677711, -0.708224, -0.659437, -0.407607, 0.677033, 0.89032, 0.228307, -0.749514, 0.772958, 0.054701, 0.551705, 0.917052, -0.895022, -0.702397, 0.484142, 0.108648, 0.833347, 0.478872, -0.984112, 0.387176, -0.73299, 0.7526, 0.443312, -0.0987856, 0.125415, 0.10876, -0.498108, 0.43209, 0.344609, 0.928941, -0.130732, -0.0569167];
    let op3_expect = [0.285357, 0.00181194, 0.453967, -0.160473, 0.133146, 0.125066, 0.695562, 0.406415, 0.612903, -0.796108, -0.221201, 0.272369, -0.181291, -0.0199411, 0.679734, 0.729573, 0.22086, 0.0192072, -0.0467102, -0.436349, 0.790771, -0.0121533, -0.102724, -0.281631, 0.146536, -0.0437044, -0.643831, -0.125283, -0.392138, 0.223089, -0.893282, -0.16027, -0.22558, -0.338964, -0.393444, 0.447179, 0.0027382, 0.0600548, 0.5614, 0.308335, -0.395642, -0.232637, -0.317546, -0.0137323, 0.0275952, -0.571289, 0.0347555, 0.609347, -0.446445, 0.27283, 0.485148, -0.602337, -0.250224, 0.551432, 0.923353, 0.360036, -0.394563, -0.64193, -0.18673, 0.796443, 0.266929, 0.421638, -0.44727, 0.926579, -0.22563, 0.663612, -0.295051, 0.44308, -0.680228, 0.36995, 0.376663, 0.654893, 0.289675, 0.107439, -0.673064, 0.0995729, 0.213019, 0.18728, -0.525372, 0.449116, -0.778254, 0.82822, 0.450766, 0.24037, 0.691436, -0.357748, 0.3905, 0.570203, 0.111496, -0.553228, 0.457363, 0.149417, -0.769431, -0.470166, -0.271529, -0.349652, 0.773931, -0.135924, 0.406866, 0.426256, -0.335963, 0.680992, -0.936889, -3.52306e-05, 0.575398, 0.509084, 0.16487, -0.657185, -0.321545, -0.338165, -0.335108, 0.902524, 0.133092, -0.790369, 0.676731, 0.46084, 0.489389, 0.66835, -0.231156, 0.0692682, -0.0377757, -0.139746, 0.105297, 0.938753, -0.403865, -0.222446, 0.45314, 0.119956, -0.388121, 0.26389, 0.27597, 0.679432, 0.700873, 0.0910737, 0.213449, 0.0917136, 0.0842865, -0.0367311, 0.214628, 0.188827, -0.243133, 0.898085, -0.271172, 0.139627, -0.319151, -0.00811307, 0.522665, -0.459861, -0.424081, -0.19957, 0.494902, -0.169442, -0.0407964, -0.629691, -0.462826, -0.567803, 0.453167, 0.0473601, 0.562038, 0.152508, 0.316812, 0.582181, 0.637157, 0.190546, -0.556541, -0.860239, -0.106728, 0.616123, -0.746842, -0.0255713, -0.453518, -0.886067, 0.418399, 0.577391, -0.467784, -0.05079, -0.685036, -0.462692, 0.460047, -0.318271, 0.708224, -0.351821, -0.364416, 0.0954479, -0.0586282, -0.0894044, 0.481278, -0.201991, -0.283279, -0.897555, 0.0611137, 0.0467872];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 8, 8, 3]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 3]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type0);

    model.setOperandValue(op2, new Float32Array([-0.966213, -0.467474, -0.82203]));
    model.setOperandValue(bias, new Float32Array([0, 0, 0]));
    model.setOperandValue(pad, new Int32Array([2]));
    model.setOperandValue(rate_w, new Int32Array([2]));
    model.setOperandValue(rate_h, new Int32Array([2]));
    model.setOperandValue(mul, new Int32Array([1]));
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

    let op3_output = new Float32Array(type0_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

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

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D vaild example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8];
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};

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
    model.addOperand(type1);

    model.setOperandValue(op2, new Float32Array([0.25, 0, 0.2, 0, 0.25, 0, 0, 0.3, 0.25, 0, 0, 0, 0.25, 0.1, 0, 0]));
    model.setOperandValue(bias, new Float32Array([1, 2, 3, 4]));
    model.setOperandValue(pad, new Int32Array([2]));
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

    let op3_output = new Float32Array(type1_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for ATROUS_DEPTHWISE_CONV_2D valid example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29];
    let op3_expect = [11, 3, 7.2, 10.9];

    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 4]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [4]};
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 4]};
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
    model.setOperandValue(pad, new Int32Array([2]));
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

    for (let i = 0; i < type4_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });

  it('check result for Concatenation axis 0 example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let tensorLength = product(float32TensorType.dimensions);

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
    compilation.setPreference(getPreferenceCode(options.prefer));
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
    let tensorLength = product(float32TensorType.dimensions);

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
    compilation.setPreference(getPreferenceCode(options.prefer));
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
    let tensorLength = product(float32TensorType.dimensions);

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
    compilation.setPreference(getPreferenceCode(options.prefer));
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
    let tensorLength = product(float32TensorType.dimensions);

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
    compilation.setPreference(getPreferenceCode(options.prefer));
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

  it('check result for Depthwise conv 28x28 input0 5x5 weights example', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op2_value = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28];

    let op3_expect = [ 348,  570,  840, 1110, 1380, 1650, 1920, 2190, 2460, 2730, 3000, 3270, 3540, 3810,
                      4080, 4350, 4620, 4890, 5160, 5430, 5700, 5970, 6240, 6510, 6780, 7050, 5580, 4137,
                       404,  660,  970, 1280, 1590, 1900, 2210, 2520, 2830, 3140, 3450, 3760, 4070, 4380,
                      4690, 5000, 5310, 5620, 5930, 6240, 6550, 6860, 7170, 7480, 7790, 8100, 6380, 4706,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       430,  700, 1025, 1350, 1675, 2000, 2325, 2650, 2975, 3300, 3625, 3950, 4275, 4600,
                      4925, 5250, 5575, 5900, 6225, 6550, 6875, 7200, 7525, 7850, 8175, 8500, 6650, 4870,
                       284,  460,  670,  880, 1090, 1300, 1510, 1720, 1930, 2140, 2350, 2560, 2770, 2980,
                      3190, 3400, 3610, 3820, 4030, 4240, 4450, 4660, 4870, 5080, 5290, 5500, 4260, 3086,
                       168,  270,  390,  510,  630,  750,  870,  990, 1110, 1230, 1350, 1470, 1590, 1710,
                      1830, 1950, 2070, 2190, 2310, 2430, 2550, 2670, 2790, 2910, 3030, 3150, 2400, 1707];

    let type0 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 1]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 28, 28, 1]};
    let type1_length = product(type1.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type3_length = product(type3.dimensions);

    let b4 = operandIndex++;
    model.addOperand(type0);
    let b5 = operandIndex++;
    model.addOperand(type0);
    let b6 = operandIndex++;
    model.addOperand(type0);
    let b7 = operandIndex++;
    model.addOperand(type0);
    let b8 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type1);
    let op0 = operandIndex++;
    model.addOperand(type2);
    let op1 = operandIndex++;
    model.addOperand(type3);

    model.setOperandValue(b4, new Int32Array([1]));
    model.setOperandValue(b5, new Int32Array([1]));
    model.setOperandValue(b6, new Int32Array([1]));
    model.setOperandValue(b7, new Int32Array([1]));
    model.setOperandValue(b8, new Int32Array([0]));

    let weights_data = new Float32Array([ 1,  2,  3,  4,  5,
                                          6,  7,  8,  9, 10,
                                         11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20,
                                         21, 22, 23, 24, 25]);
    model.setOperandValue(op0, new Float32Array(weights_data));
    model.setOperandValue(op1, new Float32Array([0]));
    model.addOperation(nn.DEPTHWISE_CONV_2D, [op2, op0, op1, b4, b5, b6, b7, b8], [op3]);

    model.identifyInputsAndOutputs([op2], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op2_input = new Float32Array(op2_value);
    execution.setInput(0, op2_input);

    let op3_output = new Float32Array(type1_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
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
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 3]};
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

  it('check result for Fully connected float 3D input example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [ 1,  2,  3,  4,  5,  6,
                      7,  8,  9, 10, 11, 12];
    let op3_expect = [8, 17, 26, 35];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 6]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
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

    model.setOperandValue(op2, new Float32Array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]));
    model.setOperandValue(b0, new Float32Array([0, 4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

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

  it('check result for Fully connected float 3D input example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [ 1,  2,  3,  4,  5,  6,
                      7,  8,  9, 10, 11, 12];
    let op3_expect = [4, 5, 10, 8, 16, 11, 22, 14];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2]};
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

    model.setOperandValue(op2, new Float32Array([1, 0, 1, 1, 0, 0]));
    model.setOperandValue(b0, new Float32Array([0, 4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

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

  it('check result for Fully connected float 4D input example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [ 1,  2,  3,  4,  5,  6,
                      7,  8,  9, 10, 11, 12,
                     13, 14, 15, 16, 17, 18,
                     19, 20, 21, 22, 23, 24];
    let op3_expect = [8, 17, 26, 35, 44, 53, 62, 71];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 6]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2]};
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

    model.setOperandValue(op2, new Float32Array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1]));
    model.setOperandValue(b0, new Float32Array([0, 4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

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

  it('check result for Fully connected float 4D input example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [ 1,  2,  3,  4,  5,  6,  7,  8,
                      9, 10, 11, 12, 13, 14, 15, 16,
                     17, 18, 19, 20, 21, 22, 23, 24];
    let op3_expect = [17, 24, 23, 49, 56, 55, 81, 88, 87];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 8]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 3]};
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

    model.setOperandValue(op2, new Float32Array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]));
    model.setOperandValue(b0, new Float32Array([1, 2, 4]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

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

  it('check result for Fully connected float 4D input example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
    let op3_expect = [40, 44, 112, 116];

    let type3 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 12]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2_length = product(type2.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 3]};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
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

    model.setOperandValue(op2, new Float32Array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1]));
    model.setOperandValue(b0, new Float32Array([1, 2]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act], [op3]);

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

  it('check result for Mul example', async function() {
    let TENSOR_DIMENSIONS = [2, 2, 2, 2];
    let value0 = 0.4;
    let value1 = 0.5;
    let operandIndex = 0;
    let model = await nn.createModel(options);
    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    let tensorLength = product(float32TensorType.dimensions);

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

    compilation.setPreference(getPreferenceCode(options.prefer));

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

  it('check result for Mul broadcasting 1D-2D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 50, 60];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 1D-2D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 50, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-2D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 50, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-2D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 60, 80, 150, 180];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-2D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 50, 60];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-2D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 20, 40];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 1D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 1D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 50, 120, 70, 160, 90, 200, 110, 240];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 50, 120, 70, 160, 90, 200, 110, 240];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 60, 80, 50, 60, 140, 160, 90, 100, 220, 240];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-3D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-3D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 90, 160, 50, 120, 210, 320, 90, 200, 330, 480];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-3D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 20, 40, 30, 40, 60, 80, 50, 60, 100, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 90, 160, 50, 120, 210, 320, 90, 200, 330, 480];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 150, 240, 210, 320, 450, 600, 550, 720];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 60, 80, 150, 180, 280, 320, 450, 500, 660, 720];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 30, 40, 100, 120, 140, 160, 270, 300, 330, 360];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 60, 80, 50, 60, 140, 160, 90, 100, 220, 240];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/7', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 40, 30, 80, 50, 120, 70, 160, 90, 200, 110, 240];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-3D example/8', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 2, 3, 4, 5, 6]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40, 50, 60]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [10, 20, 20, 40, 90, 120, 120, 160, 250, 300, 300, 360];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 1D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 1D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  40,  30,  80,  50, 120,  70, 160,
                         90, 200, 110, 240, 130, 280, 150, 320];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  40,  30,  80,  50, 120,  70, 160,
                         90, 200, 110, 240, 130, 280, 150, 320];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  60,  80,  50,  60, 140, 160,
                         90, 100, 220, 240, 130, 140, 300, 320];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  40,  90, 160,  50, 120, 210, 320,
                         90, 200, 330, 480, 130, 280, 450, 640];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 2D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4,
                                       5,  6,  7,  8]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  20,  40,  30,  40,  60,  80,
                         50,  60, 100, 120,  70,  80, 140, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  40,  90, 160,  50, 120, 210, 320,
                         90, 200, 330, 480, 130, 280, 450, 640];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  40,  30,  80, 150, 240, 210, 320,
                         90, 200, 110, 240, 390, 560, 450, 640];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  60,  80, 150, 180, 280, 320,
                         90, 100, 220, 240, 390, 420, 600, 640];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [  10,   40,   90,  160,  250,  360,  490,  640,
                          90,  200,  330,  480,  650,  840, 1050, 1280];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 3D-4D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                       25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                       37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]);

    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
                         7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12,
                        13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
                        19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
                        25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30,
                        31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36,
                        37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42,
                        43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48,
                        49, 49, 50, 50, 51, 51, 52, 52, 53, 53, 54, 54,
                        55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60];

    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/1', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [  10,   40,   90,  160,  250,  360,  490,  640,
                          90,  200,  330,  480,  650,  840, 1050, 1280];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/2', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [  10,   40,   90,  160,   50,  120,  210,  320,
                         450,  600,  770,  960,  650,  840, 1050, 1280];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/3', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [  10,   40,   30,   80,  150,  240,  210,  320,
                         450,  600,  550,  720,  910, 1120, 1050, 1280];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/4', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40,
                                       50, 60, 70, 80]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [  10,   20,   60,   80,  150,  180,  280,  320,
                         450,  500,  660,  720,  910,  980, 1200, 1280];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/5', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([ 1,  2,  3,  4,
                                        5,  6,  7,  8,
                                        9, 10, 11, 12,
                                       13, 14, 15, 16]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  50,  60,  70,  80,
                         90, 100, 110, 120, 130, 140, 150, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Mul broadcasting 4D-4D example/6', async function() {
    let model = await nn.createModel(options);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let length = product(type2.dimensions);

    let operandIndex = 0;
    let fusedActivationFuncNone = operandIndex++;
    model.addOperand({type: nn.INT32});
    model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    model.addOperand(type0);
    let input0Data = new Float32Array([1,  2,  3,  4]);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);

    model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();
    let input1Data = new Float32Array([10, 20, 30, 40]);
    execution.setInput(0, input1Data);
    let outputData = new Float32Array(length);
    execution.setOutput(0, outputData);
    await execution.startCompute();

    let expectedData = [ 10,  20,  30,  40,  20,  40,  60,  80,
                         30,  60,  90, 120,  40,  80, 120, 160];
    for (let i = 0; i < length; ++i) {
      assert.isTrue(almostEqualCTS(outputData[i], expectedData[i]));
    }
  });

  it('check result for Reshape example', async function() {
    let model = await nn.createModel(options);

    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions:[1, 4]};
    let tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
    model.setOperandValue(1, new Int32Array([2, 2]));
    model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
    model.addOperation(nn.RESHAPE, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 4.5, 7, 6, 10, 6, 10, 7, 8, 8.5, 11, 10, 14, 10, 14, 9, 10, 10.5, 13, 12, 16, 12, 16, 3, 4, 4.5, 7, 6, 10, 6, 10, 7, 8, 8.5, 11, 10, 14, 10, 14, 9, 10, 10.5, 13, 12, 16, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4, 2]};
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
    model.setOperandValue(width, new Int32Array([4]));
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/10', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([4]));
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

  it('check result for Resize bilinear by align_corners(FALSE) distorted example/11', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7,
                      1, 3, 3, 5, 5, 7, 5, 7];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([4]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) distorted example/12', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1,  3, 2.3333335, 4.3333335, 3.6666667, 5.6666667,  5,  7,
                      9, 11, 10.333333, 12.333333, 11.666667, 13.666667, 13, 15,
                      1,  3, 2.3333335, 4.3333335, 3.6666667, 5.6666667,  5,  7,
                      9, 11, 10.333333, 12.333333, 11.666667, 13.666667, 13, 15];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
    let op2_expect = [1, 3, 9, 11, 13, 15, 9, 11, 1, 3, 9, 11, 13, 15, 9, 11];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 6.3333335, 8.333334, 11.666667, 13.666667,1, 3, 6.3333335, 8.333334, 11.666667, 13.666667,1, 3, 6.3333335, 8.333334, 11.666667, 13.666667,1, 3, 6.3333335, 8.333334, 11.666667, 13.666667,1, 3, 6.3333335, 8.333334, 11.666667, 13.666667,1, 3, 6.3333335, 8.333334, 11.666667, 13.666667];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334, 1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

  it('check result for Resize bilinear by align_corners(FALSE) distorted example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334, 1, 3, 3.6666667, 5.666667, 5, 7, 6.333333, 8.333333, 9, 11, 10.333333, 12.333333, 6.333334, 8.333334, 9.000001, 11.000001, 10.333334, 12.333334];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) distorted example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 3, 5, 5, 7, 5, 7, 7, 9, 9, 11, 9, 11, 11, 13, 13, 15, 1, 3, 3, 5, 5, 7, 5, 7, 7, 9, 9, 11, 9, 11, 11, 13, 13, 15];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
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

  it('check result for Resize bilinear with inputs (without align_corners) distorted example/7', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([4]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear by align_corners(FALSE) distorted example/8', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
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
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) distorted example/9', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15,
                     1, 3, 5, 7, 9, 11, 13, 15];
    let op2_expect = [1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15,
                      1, 3, 13, 15];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]};
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
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) remain size example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) remain size example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) remain size example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) remain size example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear by align_corners(FALSE) remain size example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) remain size example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];
    let op2_expect = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom in example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16, 3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

  it('check result for Resize bilinear by align_corners(FALSE) zoom in example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16, 3, 4, 5, 8, 6, 10, 7, 8, 9, 12, 10, 14, 9, 10, 11, 14, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) zoom in example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3, 4, 4.5, 7, 6, 10, 6, 7, 7.5, 10, 9, 13, 9, 10, 10.5, 13, 12, 16, 3, 4, 4.5, 7, 6, 10, 6, 7, 7.5, 10, 9, 13, 9, 10, 10.5, 13, 12, 16];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
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

    model.setOperandValue(height, new Int32Array([3]));
    model.setOperandValue(width, new Int32Array([3]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom in example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3,  4,  4.5,  7,  6, 10,  6, 10,
                      6,  7,  7.5, 10,  9, 13,  9, 13,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      3,  4,  4.5,  7,  6, 10,  6, 10,
                      6,  7,  7.5, 10,  9, 13,  9, 13,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      9, 10, 10.5, 13, 12, 16, 12, 16];

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

    model.setOperandValue(height, new Int32Array([4]));
    model.setOperandValue(width, new Int32Array([4]));
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

  it('check result for Resize bilinear by align_corners(FALSE) zoom in example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [3, 4, 6, 10, 9, 10, 12, 16, 3, 4, 6, 10, 9, 10, 12, 16];
    let op2_expect = [3,  4,  4.5,  7,  6, 10,  6, 10,
                      6,  7,  7.5, 10,  9, 13,  9, 13,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      3,  4,  4.5,  7,  6, 10,  6, 10,
                      6,  7,  7.5, 10,  9, 13,  9, 13,
                      9, 10, 10.5, 13, 12, 16, 12, 16,
                      9, 10, 10.5, 13, 12, 16, 12, 16];
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
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 4, 10, 13];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 4, 10, 13, 1, 4, 10, 13];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 1]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 7, 9, 10, 12, 11.5, 9];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/4', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 7, 9, 10, 12, 11.5, 9, 1, 3, 7, 9, 10, 12, 11.5, 9];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear by align_corners(FALSE) zoom out example/5', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 7, 9, 10, 12, 11.5, 9, 1, 3, 7, 9, 10, 12, 11.5, 9];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) zoom out example/6', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17, 1, 3, 5, 7, 9, 11, 13, 15, 17];
    let op2_expect = [1, 3, 9, 11, 7, 9, 15, 17, 1, 3, 9, 11, 7, 9, 15, 17];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 3, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear with inputs (without align_corners) zoom out example/7', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
    let op2_expect = [1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    let type1_length = product(type1.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let height = operandIndex++;
    model.addOperand(type2);
    let width = operandIndex++;
    model.addOperand(type2);

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Resize bilinear by align_corners(FALSE) zoom out example/8', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
    let op2_expect = [1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11,
                      1, 3, 9, 11];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
    model.setOperandValue(align_corners, new Int32Array([0]));
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

  it('check result for Resize bilinear by align_corners(TRUE) zoom out example/9', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
    let op2_expect = [1, 3, 13, 15, 17, 19, 29, 31,
                      1, 3, 13, 15, 17, 19, 29, 31];

    let type2 = {type: nn.INT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 4, 2]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
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

    model.setOperandValue(height, new Int32Array([2]));
    model.setOperandValue(width, new Int32Array([2]));
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

  it('check result for Softmax with 2D input tensor example', async function() {
    let model = await nn.createModel(options);
    let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    let tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.FLOAT32});
    model.setOperandValue(1, new Float32Array([1.0]));
    model.addOperand(float32TensorType);
    model.addOperation(nn.SOFTMAX, [0, 1], [2]);

    model.identifyInputsAndOutputs([0], [2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
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

  it('check result for Softmax float with 4D input tensor example-1', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input_value = [10.63, 18.75, 12.91, 9.46, 7.31, 12.48, 9.55, 14.28];
    let output_expect = [0.0002926, 0.9835261, 0.0028609, 0.0000908, 0.0000106, 0.0018611, 0.0000994, 0.0112587];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 8]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type0_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('check result for Softmax float with 4D input tensor example-2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input_value = [10.63, 18.75, 12.91, 9.46, 7.31, 12.48, 9.55, 14.28, 19.07, 15.91, 18.47, 20.08];
    let output_expect = [0.000296 , 0.9948254, 0.0028938, 0.0000919, 0.0000107, 0.0018824, 0.0000169, 0.001913 , 0.2301376, 0.0097638, 0.1263021, 0.6318661];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 6]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type0_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });

  it('check result for Softmax float with 4D input tensor example-3', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input_value = [
      10.633096694946289,18.750228881835938,12.917834281921387,9.46863079071045,7.316360950469971,12.48576545715332,9.552565574645996,14.280917167663574,19.07547378540039,15.915868759155273,18.47535514831543,20.08603286743164,19.306753158569336,16.763883590698242,17.23187828063965,13.828563690185547,16.604629516601562,15.456416130065918,12.026884078979492,17.517009735107422,13.086971282958984,17.17919158935547,13.359915733337402,18.612995147705078,15.037846565246582,12.749266624450684,20.888051986694336,19.03054428100586,12.739729881286621,20.859451293945312,9.963472366333008,17.446157455444336,13.153258323669434,10.8407621383667,7.386678695678711,12.040700912475586,14.887235641479492,8.948171615600586,14.241692543029785,13.147953987121582,19.639558792114258,16.8834171295166,20.867238998413086,15.453288078308105,15.594100952148438,8.695173263549805,19.68443489074707,20.516754150390625,12.195821762084961,7.397640705108643,10.379465103149414,7.701779842376709,14.675374984741211,12.548005104064941,13.878507614135742,10.712503433227539,10.09022331237793,9.910835266113281,13.521472930908203,11.991893768310547,10.604318618774414,13.52990436553955,12.962638854980469,10.640522956848145,11.012723922729492,10.303203582763672,17.868167877197266,12.640779495239258,16.51861572265625,8.668283462524414,14.46950626373291,11.660629272460938,13.119953155517578,20.012876510620117,17.966630935668945,14.317724227905273,10.995388984680176,21.0853271484375,20.428958892822266,14.6182279586792,16.351455688476562,13.798954010009766,18.01214027404785,16.112661361694336,16.145004272460938,21.265243530273438,19.746318817138672,15.0040922164917,18.015207290649414,13.846283912658691,14.229297637939453,21.80757713317871,20.887664794921875,17.63591766357422,21.04983139038086,18.448760986328125,15.622299194335938,13.906994819641113,10.934183120727539,17.315757751464844,13.339083671569824,4.85207986831665,13.862253189086914,17.0489559173584,17.090391159057617,19.222625732421875,18.432119369506836,12.8416748046875,13.289155960083008,5.505262851715088,14.440519332885742,11.779170036315918,13.61559772491455,17.74374008178711,15.418974876403809,16.40049934387207,10.327967643737793,10.13575267791748,8.87507438659668,11.005951881408691,15.417574882507324,9.246169090270996,11.587608337402344,7.928442001342773,12.507555961608887,14.174178123474121,14.090804100036621,12.676902770996094,12.651900291442871,12.657724380493164,14.737821578979492,10.923689842224121,10.040297508239746,14.572811126708984,18.230188369750977,14.233528137207031,17.16696548461914,12.539831161499023,17.216655731201172,10.261114120483398,15.773637771606445,9.769366264343262,8.743120193481445,7.953049659729004,16.061744689941406,12.244834899902344,9.669194221496582,6.421072006225586,7.566117286682129,10.127528190612793,16.603940963745117,23.608600616455078,18.561338424682617,18.106245040893555,16.091753005981445,14.153021812438965,11.953116416931152,17.563020706176758,17.049118041992188,9.418050765991211,9.063760757446289,5.397731781005859,11.469522476196289,5.9847636222839355,5.454802989959717,5.276651859283447,8.385995864868164,10.397110939025879,9.794140815734863,7.765122413635254,9.706226348876953,12.75146484375,11.481279373168945,15.583362579345703,12.664912223815918,12.48148250579834,9.0400972366333,10.423805236816406,9.941632270812988,14.847150802612305,11.887077331542969,13.975244522094727,18.85527801513672,15.132763862609863,15.222917556762695,21.931215286254883,22.93506622314453,16.561208724975586,15.666217803955078,16.318681716918945,10.93188762664795,9.766206741333008,18.415191650390625,15.500444412231445,13.569294929504395,14.135770797729492,15.194920539855957,9.175687789916992,12.3844633102417,12.382109642028809,14.051812171936035,13.496564865112305,16.24966049194336,18.620302200317383,17.311433792114258,11.768840789794922,10.057878494262695,16.888906478881836,13.056600570678711,10.114962577819824,6.360996723175049,13.598278999328613,10.554468154907227,10.785820960998535,11.084970474243164,13.954259872436523,9.376203536987305,9.118735313415527,9.517805099487305,13.780681610107422,10.723291397094727,9.961705207824707,13.04687213897705,16.338111877441406,14.786600112915039,10.975630760192871,7.872645854949951,12.923849105834961,8.248885154724121,8.072661399841309,13.284008979797363,17.742828369140625,19.021516799926758,12.659760475158691,10.832762718200684,12.244898796081543,9.773988723754883,11.93078327178955,9.257373809814453,12.855767250061035,11.640983581542969,6.922642230987549,10.941132545471191,6.090719223022461,9.144007682800293,14.85858154296875,5.732938289642334,7.951379776000977,22.1893253326416,18.594200134277344,19.11495018005371,9.753591537475586,17.731361389160156,16.950284957885742,12.792081832885742,8.655010223388672,11.664034843444824,13.184113502502441,17.379167556762695,21.497314453125,18.083168029785156,13.747838973999023,16.312685012817383,22.464080810546875,17.72537612915039,21.2579402923584,19.709409713745117,17.889324188232422,12.99542236328125,19.886795043945312,19.78012466430664,18.930387496948242,20.853300094604492,20.380077362060547,20.485424041748047,11.897425651550293,21.703922271728516,25.7706298828125,25.417396545410156,22.408151626586914,21.254987716674805,21.25147247314453,24.951309204101562,19.646259307861328,15.995959281921387,20.60489273071289,22.796550750732422,25.586605072021484,17.506654739379883,18.077157974243164,17.154869079589844,16.19427490234375,19.657270431518555,22.114973068237305,22.831348419189453,16.289579391479492,16.08236312866211,12.500151634216309,27.87720489501953,22.74942398071289,11.590218544006348,23.27147102355957,15.392729759216309,16.297754287719727,19.246252059936523,13.171932220458984,18.263992309570312,19.38363265991211,16.606826782226562,21.869565963745117,21.531877517700195,14.4906644821167,17.118247985839844,13.3585786819458,13.36583137512207,14.172847747802734,14.482640266418457,20.355361938476562,22.65353012084961,20.236610412597656,22.6688232421875,12.264982223510742,22.149799346923828,10.919144630432129,16.730859756469727,18.9865779876709,17.438749313354492,12.164246559143066,9.182199478149414,13.177031517028809,16.747133255004883,22.318220138549805,16.13444709777832,31.314706802368164,14.171764373779297,20.046432495117188,19.96190071105957,19.90066909790039,19.768455505371094,9.5852689743042,14.080363273620605,18.218555450439453,16.570444107055664,11.141199111938477,8.320511817932129,9.214081764221191,6.724745273590088,10.460344314575195,10.91906452178955,10.70103645324707,8.582121849060059,5.750553131103516,8.196184158325195,11.705013275146484,11.400447845458984,15.86993408203125,29.46346664428711,28.481582641601562,30.271581649780273,27.074277877807617,23.513687133789062,14.259750366210938,22.21346664428711,10.838545799255371,13.458712577819824,14.445221900939941,10.288758277893066,9.76822566986084,16.683469772338867,15.019920349121094,17.477670669555664,21.0462646484375,11.625995635986328,17.343097686767578,20.955812454223633,14.320205688476562,13.160045623779297,23.845062255859375,17.921173095703125,18.691030502319336,25.263710021972656,23.509864807128906,21.79447364807129,17.10430908203125,20.87082290649414,6.691738128662109,4.533743858337402,32.7180290222168,14.728958129882812,12.593729019165039,11.756786346435547,12.521442413330078,13.999703407287598,7.966588020324707,7.5634331703186035,12.278633117675781,8.848836898803711,21.33017349243164,15.9844331741333,9.855817794799805,7.322624206542969,9.465375900268555,9.09567928314209,5.431189060211182,11.508605003356934,9.423563003540039,5.644067287445068,9.203815460205078,6.727909564971924,12.500838279724121,8.945449829101562,7.68461275100708,11.250411033630371,7.7577900886535645,7.69315242767334,18.22950553894043,11.466642379760742,11.124119758605957,17.135154724121094,20.294755935668945,7.910463333129883,12.207856178283691,10.171067237854004,11.036746978759766,15.994007110595703,9.576361656188965,10.139100074768066,11.252339363098145,19.56910514831543,15.305747985839844,8.877645492553711,10.725064277648926,11.874349594116211,7.23990535736084,23.156518936157227,12.889435768127441,7.595884799957275,12.746710777282715,15.353927612304688,13.408821105957031,15.109270095825195,12.982090950012207,8.926817893981934,20.238067626953125,5.082365989685059,9.151376724243164,14.556602478027344,14.110031127929688,12.146501541137695,9.675328254699707,11.475993156433105,10.607942581176758,20.15665626525879,16.725154876708984,14.798490524291992,14.677826881408691,17.104942321777344,22.826847076416016,6.6239824295043945,11.857154846191406,11.215736389160156,9.413376808166504,13.951075553894043,20.70792007446289,13.43892765045166,7.521987438201904,8.435393333435059,7.710779190063477,7.7452778816223145,11.229757308959961,18.31511878967285,12.494784355163574,7.136443614959717,13.607855796813965,11.863105773925781,15.43836784362793,10.102659225463867,9.572657585144043,20.003894805908203,7.735821723937988,10.727670669555664,14.5988187789917,8.454527854919434,13.764562606811523,9.13848876953125,11.05479621887207,7.651605129241943,19.541263580322266,13.013172149658203,18.636497497558594,8.763279914855957,10.759872436523438,5.887054920196533,6.702114582061768,14.914555549621582,7.584196090698242,18.391887664794922,11.342570304870605,16.591960906982422,9.99575138092041,2.692444086074829,11.62830924987793,12.326775550842285,10.393973350524902,18.7794246673584,11.606731414794922,16.985645294189453,12.819670677185059,22.42898178100586,17.77494239807129,7.2856574058532715,5.674855709075928,16.14600944519043,9.735804557800293,7.700339317321777,16.8890323638916,12.788625717163086,8.340590476989746,9.897276878356934,12.495906829833984,16.799942016601562,11.413158416748047,12.565237998962402,13.016317367553711,6.6281514167785645,8.394243240356445,11.428860664367676,12.286535263061523,12.476916313171387,13.393653869628906,18.071990966796875,13.326603889465332,14.388191223144531,16.641151428222656,20.775880813598633,8.633672714233398,6.971398830413818,15.458163261413574,9.286026000976562,20.402040481567383,6.572159290313721,11.029325485229492,15.130033493041992,19.171951293945312,8.491789817810059,13.256662368774414,10.625027656555176,4.617883682250977,9.81132698059082,14.932689666748047,8.410503387451172,11.563192367553711,15.107598304748535,8.680805206298828,7.17470121383667,10.41996955871582,10.600149154663086,11.872647285461426,11.262316703796387,13.336587905883789,6.557470321655273,6.086480140686035,17.123538970947266,12.54118537902832,6.883083820343018,7.150421619415283,11.645964622497559,12.30489444732666,11.739986419677734,7.104118347167969,13.120207786560059,7.805128574371338,15.813161849975586,5.132405757904053,11.732964515686035,12.729701042175293,7.854181289672852,10.323857307434082,10.044418334960938,7.791658878326416,12.905342102050781,6.566869735717773,16.58338165283203,6.539013385772705,19.956323623657227,15.6813325881958,6.952506065368652,15.236894607543945,19.901325225830078,12.898874282836914,13.024730682373047,15.449207305908203,13.418533325195312,13.35937213897705,6.902468204498291,9.149248123168945,12.655171394348145,6.908873081207275,7.920547962188721,18.209186553955078,16.290029525756836,8.436335563659668,15.455425262451172,10.462504386901855,11.550963401794434,14.850924491882324,14.157072067260742,6.95953369140625,13.412869453430176,9.158906936645508,7.781056880950928,16.6036434173584,4.808084964752197,16.464162826538086,8.920628547668457,7.468282222747803,15.928406715393066,10.48175048828125,15.1595458984375,16.93899154663086,20.6048641204834,10.1943998336792,23.46732521057129,14.569906234741211,13.991246223449707,9.545177459716797,14.73549747467041,4.1087470054626465,8.822433471679688,10.808523178100586,6.976772785186768,13.555315017700195,11.995615005493164,17.68769645690918,7.434670448303223,11.686838150024414,11.780466079711914,13.016263008117676,11.471842765808105,8.670991897583008,6.094179630279541,13.375377655029297,4.240854263305664,14.909464836120605,16.622299194335938,12.553129196166992,4.400779724121094,14.271684646606445,15.330678939819336,5.092896938323975,17.178447723388672,11.311455726623535,8.151897430419922,13.499678611755371,8.588479042053223,9.777791976928711,6.631325721740723,7.920196056365967,18.568910598754883,11.4961519241333,9.567403793334961,3.42606520652771,13.4708833694458,7.893254280090332,16.88814926147461,7.639266014099121,11.274413108825684,12.329423904418945,5.295083045959473,8.608922004699707,11.687641143798828,5.501578330993652,8.556900978088379,22.424365997314453,21.320537567138672,9.413985252380371,16.24306297302246,18.76443862915039,12.455265998840332,9.97346305847168,19.657798767089844,19.776670455932617,14.17664909362793,13.859002113342285,11.102787971496582,5.418384552001953,13.521247863769531,5.904786586761475,11.003355979919434,9.2664155960083,10.399650573730469,8.900339126586914,18.227764129638672,11.308398246765137,8.56893253326416,15.272917747497559,21.683618545532227,8.110478401184082,8.979012489318848,14.438859939575195,23.816020965576172,10.164922714233398,12.940319061279297,15.51875114440918,14.825761795043945,9.596190452575684,8.136138916015625,12.112674713134766,8.235942840576172,14.2014799118042,15.066385269165039,14.520268440246582,15.384756088256836,15.410542488098145,13.188827514648438,8.248003005981445,20.823171615600586,5.83102560043335,8.386247634887695,15.35638427734375,12.13347053527832,12.067032814025879,23.452890396118164,17.4442195892334,6.897007465362549,16.249801635742188,7.627880573272705,8.791763305664062,18.85394859313965,11.848384857177734,7.161445617675781,14.167245864868164,9.618569374084473,13.03767204284668,4.883429527282715,12.788015365600586,9.504534721374512,16.950599670410156,17.135223388671875,5.525537014007568,11.9367036819458,8.002015113830566,19.02523422241211,9.852280616760254,10.827128410339355,6.175796985626221,7.862109184265137,10.792503356933594,14.547138214111328,19.57797622680664,10.705717086791992,5.619636535644531,12.487151145935059,12.44333553314209,13.876189231872559,8.839409828186035,8.618517875671387,7.086108684539795,13.258471488952637,8.735761642456055,15.709990501403809,23.941495895385742,14.898078918457031,11.989447593688965,9.777897834777832,15.606669425964355,8.85151481628418,20.076242446899414,10.302022933959961,19.189739227294922,10.508451461791992,9.3486328125,16.8221378326416,23.214120864868164,7.9513139724731445,9.37641716003418,10.079035758972168,10.692888259887695,15.500971794128418,8.896175384521484,7.747472763061523,11.700419425964355,17.939682006835938,16.658222198486328,18.93973159790039,14.567663192749023,9.807376861572266,10.250672340393066,11.922234535217285,9.841506004333496,17.176061630249023,15.122502326965332,11.642127990722656,9.288005828857422,17.532371520996094,11.409940719604492,10.133581161499023,17.396413803100586,10.805451393127441,13.814460754394531,10.8898344039917,6.822375297546387,7.756640911102295,5.883997917175293,18.50943374633789,15.444413185119629,13.512382507324219,10.010627746582031,13.188455581665039,16.911428451538086,14.24902629852295,17.83585548400879,7.323952674865723,17.830869674682617,10.343512535095215,15.469857215881348,14.64986515045166,4.747836589813232,14.0709810256958,14.859786033630371,11.047195434570312,6.187945365905762,6.384342193603516,15.852181434631348,17.95515251159668,13.279470443725586,15.942207336425781,12.031232833862305,17.550067901611328,9.142046928405762,7.987494468688965,10.954753875732422,9.919310569763184,6.418801307678223,9.850926399230957,12.471698760986328,14.544173240661621,14.931924819946289,13.729412078857422,7.002838611602783,18.513202667236328,14.387741088867188,6.659890174865723,12.94911003112793,15.653473854064941,20.65776252746582,11.097545623779297,8.67094612121582,11.96640682220459,14.012194633483887,23.167497634887695,12.37126350402832,22.66785430908203,10.08847427368164,7.214731693267822,20.058713912963867,11.182188987731934,8.85742473602295,10.458805084228516,15.855852127075195,12.43456745147705,17.035202026367188,14.466036796569824,7.409712791442871,4.4751296043396,17.218917846679688,10.063966751098633,10.104472160339355,13.607396125793457,6.362962245941162,13.306111335754395,4.225671768188477,16.492111206054688,10.005599975585938,3.6910488605499268,8.502364158630371,17.432174682617188,7.754965782165527,13.66734504699707,18.698524475097656,9.998183250427246,10.519367218017578,10.32975959777832,16.7487735748291,11.253996849060059,19.485973358154297,8.824007987976074,6.66616153717041,5.742675304412842,10.295378684997559,10.892179489135742,8.883288383483887,12.90867805480957,11.247989654541016,6.21877908706665,11.02491569519043,18.458473205566406,17.864404678344727,19.68421745300293,15.547470092773438,10.0148344039917,11.02116584777832,18.050172805786133,14.298063278198242,23.68663787841797,14.349076271057129,8.816232681274414,13.859445571899414,9.727208137512207,14.893916130065918,22.43604850769043,17.441837310791016,19.600698471069336,7.264801502227783,2.443992853164673,4.433985233306885,15.922985076904297,11.761265754699707,16.840246200561523,10.411992073059082,5.234299659729004,12.049160957336426,12.798580169677734,15.829668045043945,14.996968269348145,16.621143341064453,8.305027961730957,10.319125175476074,11.589641571044922,24.855915069580078,11.700460433959961,10.665645599365234,6.567962646484375,8.83480167388916,13.192261695861816,13.800191879272461,10.726040840148926,23.260066986083984,14.184962272644043,15.241336822509766,12.362273216247559,9.456809997558594,15.895992279052734,17.549957275390625,13.98581600189209,18.19790267944336,12.856616020202637,22.54080581665039,20.437210083007812,18.296781539916992,20.601491928100586,17.888147354125977,21.95772933959961,17.490028381347656,19.035091400146484,10.328767776489258,17.410137176513672,15.300915718078613,15.674595832824707,7.31687593460083,6.946949005126953,11.746861457824707,10.91621208190918,10.711849212646484,17.64960289001465,16.84800910949707,13.10483455657959,10.787004470825195,19.537708282470703,13.479764938354492,10.648937225341797,20.14699363708496,7.483791351318359,12.170218467712402,11.080985069274902,18.158065795898438,6.356677055358887,9.530189514160156,13.072765350341797,5.075244903564453,10.3015718460083,10.833891868591309,12.011783599853516,5.759989261627197,18.32382583618164,21.10507583618164,10.329180717468262,24.04796600341797,18.622648239135742,18.175914764404297,20.407981872558594,11.882205963134766,16.208574295043945,9.092352867126465,13.63718318939209,15.504472732543945,15.205802917480469,14.812711715698242,25.147401809692383,18.61565399169922
    ];
    let output_expect = [
      1.8249300681727476e-10,6.116051167737169e-7,1.7926399253909153e-9,5.695406299355277e-11,6.619194775625559e-12,1.1637220076465837e-9,6.194108420620736e-11,7.006050850577594e-9,8.466907956972136e-7,3.593595110373826e-8,4.646186653189943e-7,0.0000023259663066710345,0.0000010670113397281966,8.39102085592458e-8,1.3398664577835007e-7,4.456800706265085e-9,7.155678360959428e-8,2.2698046109326242e-8,7.354650777280369e-10,1.781944547474268e-7,2.123014208876839e-9,1.271098142296978e-7,2.789265929337148e-9,5.331788770490675e-7,1.4935055858700252e-8,1.5145609211231204e-9,0.000005187015631236136,8.094943950709421e-7,1.500185531355669e-9,0.000005040726136940066,9.341904333437512e-11,1.6600490937435097e-7,2.268513821235274e-9,2.2461242588089902e-10,7.101400197051522e-12,7.456959494334114e-10,1.2846859576143288e-8,3.38449754333503e-11,6.736574409416107e-9,2.2565083135361874e-9,0.0000014883429457768216,9.456402239038653e-8,0.000005080212758912239,2.2627158813293136e-8,2.604856774723885e-8,2.6279480327962013e-11,0.0000015566582760584424,0.00000357821659235924,8.708197496432035e-10,7.179711252608412e-12,1.4161076478114154e-10,9.73176747520732e-12,1.0394024307913696e-8,1.2384566705492261e-9,4.685007493065996e-9,1.9757658009655898e-10,1.0604274403025471e-10,8.862839628864805e-11,3.278344262724886e-9,7.101745858051345e-10,1.7731742463222844e-10,3.3060865156642194e-9,1.874797650458504e-9,1.8385312716695523e-10,2.667577825299361e-10,1.3121283226613656e-10,2.5316251139884116e-7,1.3588561387223308e-9,6.565893784227228e-8,2.558234353522426e-11,8.460118827713359e-9,5.099181077383719e-10,2.1941859440488543e-9,0.0000021618986920657335,2.793586872940068e-7,7.268775803481731e-9,2.621741435060443e-10,0.0000063181955738400575,0.0000032774305509519763,9.816756296743279e-9,5.5551996780422996e-8,4.326729641235261e-9,2.923641773122654e-7,4.3751381184620186e-8,4.5189647579491066e-8,0.0000075636212386598345,0.0000016560430822210037,1.4439318185566208e-8,2.9326221806513786e-7,4.536449882408533e-9,6.653596340555623e-9,0.000013009574104216881,0.000005184987912798533,2.0069361994501378e-7,0.000006097848654462723,4.5242711621540366e-7,2.6793543383973883e-8,4.820421395379526e-9,2.466070958551114e-10,1.4571017459275026e-7,2.731750603501837e-9,5.630991007678221e-13,4.609471027094969e-9,1.1158847001979666e-7,1.1630885410340852e-7,9.809130006033229e-7,4.4496027840068564e-7,1.6611939601673953e-9,2.5987216822898063e-9,1.0820738836209154e-12,8.21843038067982e-9,5.740911079854527e-10,3.6018896754086427e-9,2.235442906339813e-7,2.1863918675535388e-8,5.8344269859844644e-8,1.3450288105509856e-10,1.1098229280026572e-10,3.145914084790036e-11,2.6495813876259433e-10,2.1833246321989463e-8,4.559458713360165e-11,4.740103864975254e-10,1.2207628137528292e-11,1.189361720221882e-9,6.296795529436849e-9,5.793106883089649e-9,1.4088370470233258e-9,1.3740559801078689e-9,1.3820726785240822e-9,1.1063875149375235e-8,2.4403454257360124e-10,1.008783681588632e-10,9.380873855491245e-9,3.635971097537549e-7,6.681817321663175e-9,1.2556522221984778e-7,1.2283783989985864e-9,1.319624658435714e-7,1.2580547714691193e-10,3.117129665497487e-8,7.693680414977422e-11,2.7570249902519883e-11,1.2511791082359114e-11,4.157962862905151e-8,9.145715851310854e-10,6.960345494411158e-11,2.7038681623542393e-12,8.49717171624853e-12,1.1007285360964403e-10,7.150657665988547e-8,0.00007878302858443931,5.063358798906847e-7,3.2121070603352564e-7,4.284635224394151e-8,6.164954768905773e-9,6.831636922832729e-10,1.8658418809991417e-7,1.1160614121763501e-7,5.414501833000962e-11,3.799215803068634e-11,9.717554452087773e-13,4.212125648273002e-10,1.7478408043808402e-12,1.0288263296853728e-12,8.609374606456666e-13,1.9290504957303334e-11,1.4413163718085542e-10,7.886622604980076e-11,1.0368188946285883e-11,7.222899361947199e-11,1.5178924783754155e-9,4.261972164076866e-10,2.5770294342919442e-8,1.3920411490175866e-9,1.158753981655991e-9,3.710357715513979e-11,1.4803136494379032e-10,9.140060930334926e-11,1.2342080246696696e-8,6.395098894884654e-10,5.160866400899522e-9,6.793536044824577e-7,1.642214009223153e-8,1.7971462540344874e-8,0.00001472174517402891,0.00004017187166027725,6.851595912849007e-8,2.799650111739993e-8,5.3760949469960906e-8,2.460428250028457e-10,7.66941718466363e-11,4.37488182569723e-7,2.3719570307889626e-8,3.4389011638324973e-9,6.0595204409708e-9,1.7475176861125874e-8,4.2491617863182896e-11,1.051609910263096e-9,1.049139664033305e-9,5.5715374536191575e-9,3.1976714609527335e-9,5.017546556018715e-8,5.370870326260047e-7,1.4508121637391014e-7,5.681911607879897e-10,1.0266749256304664e-10,9.508508469480148e-8,2.0594983496380337e-9,1.0869920241685094e-10,2.546228635921799e-12,3.5400531395168855e-9,1.6869379504402815e-10,2.126052528472755e-10,2.867427961739111e-10,5.053676588318012e-9,5.192605373460779e-11,4.013919752132722e-11,5.982467993215224e-11,4.248412288632153e-9,1.9971842235566584e-10,9.325347438693399e-11,2.0395654054539136e-9,5.481511777816195e-8,1.1616889672438901e-8,2.5704485762112483e-10,1.1545171069160531e-11,1.8034636006802884e-9,1.681890043903067e-11,1.410142228680522e-11,2.5853830187827498e-9,2.2333928484385979e-7,8.022150836950459e-7,1.384896197720309e-9,2.228236761769864e-10,9.146239321466965e-10,7.729304002390691e-11,6.68074318088685e-10,4.610838100216341e-11,1.6847684358722859e-9,4.999962666119018e-10,4.464962254091542e-12,2.4832849665479273e-10,1.9432077474901277e-12,4.1166538927717156e-11,1.2483943656604879e-8,1.358734739304035e-12,1.2490904144346615e-11,0.000019056877135881223,5.232483886175032e-7,8.807802487353911e-7,7.573230931257058e-11,2.2079163386479195e-7,1.0110312587130466e-7,1.5808230280356383e-9,2.524497971778672e-11,5.116571610841447e-10,2.3395780868185057e-9,1.552485571210127e-7,0.000009539289749227464,3.1388600518766907e-7,4.1111416493322395e-9,5.343921927192241e-8,0.000025082830688916147,2.194749129103002e-7,0.000007508597263949923,0.0000015960250721036573,2.5857605123746907e-7,1.9372785597937536e-9,0.0000019057885083384463,0.0000017129677871707827,7.323390605051827e-7,0.000005009815140510909,0.000003121077043033438,0.0000034678289466683054,6.46155295935813e-10,0.000011728605386451818,0.0006845314055681229,0.0004808211815543473,0.00002371838672843296,0.000007486389222322032,0.000007460118922608672,0.0003016941773239523,0.0000014983462506279466,3.893206468319477e-8,0.000003907889094989514,0.00003497596480883658,0.0005694731371477246,1.7635809967941896e-7,3.1200636385619873e-7,1.240557310211443e-7,4.747210979871852e-8,0.0000015149446426221402,0.000017691450921120122,0.00003621419455157593,5.221862764415164e-8,4.2445428505288874e-8,1.1805924016172753e-9,0.005626885686069727,0.00003336576992296614,4.752487847703435e-10,0.0000562373643333558,2.129764098413034e-8,5.264745794875125e-8,0.0000010043719385066652,2.3112716185380577e-9,3.7609891023748787e-7,0.00000115226941943547,7.171418303641985e-8,0.000013841569852957036,0.000009874836905510165,8.641044324519953e-9,1.195948584609141e-7,2.785533803617568e-9,2.8058180223666795e-9,6.288394249764906e-9,8.57198401149617e-9,0.0000030448723009612877,0.00003031481901416555,0.000002703938207559986,0.00003078205190831795,9.331846406723798e-10,0.000018318361981073394,2.4292695632865957e-10,8.118396266354466e-8,7.746661481178307e-7,1.6477966369166097e-7,8.437586740406289e-10,4.2769048719248914e-11,2.3230721790667985e-9,8.251655003732594e-8,0.000021678664779756218,4.471499437386228e-8,0.17504729330539703,6.281609454816817e-9,0.000002235674173789448,0.000002054448032140499,0.000001932413852046011,0.0000016931240907069878,6.400025648334307e-11,5.732947894188101e-9,3.5939260101258697e-7,6.915166039789256e-8,3.03327862827274e-10,1.8067875320881832e-11,4.415480481245737e-11,3.6632979635853324e-12,1.5353945892471188e-10,2.4290841560414833e-10,1.953231743012651e-10,2.3470440868589293e-11,1.3828879447813636e-12,1.5955433155245835e-11,5.330600960640197e-10,3.9310335542275254e-10,3.4322450659374226e-8,0.027489984408020973,0.01029787678271532,0.061678461730480194,0.002520942594856024,0.00007164949056459591,6.859346424192836e-9,0.00001952238380908966,2.2411517086595723e-10,3.078891364083347e-9,8.257208250483927e-9,1.2933092097267007e-10,7.684909653082883e-11,7.742669083654619e-8,1.4669755188378986e-8,1.7132053642399114e-7,0.000006076161298551597,4.925601038152649e-10,1.4974969531067472e-7,0.0000055506720855191816,7.286835579378703e-9,2.283947919679008e-9,0.00009980044706026092,2.6694192456488963e-7,5.764510433436953e-7,0.00041232851799577475,0.00007137696229619905,0.000012840313502238132,1.1793984810992697e-7,0.000005098384463053662,3.544358166338024e-12,4.0957539551751687e-13,0.7122098803520203,1.096618085227874e-8,1.2963921047770555e-9,5.613841058682567e-10,1.2059825360566379e-9,5.288656623747556e-9,1.2682308328515468e-11,8.474383521306361e-12,9.460036087816093e-10,3.0644525111922505e-11,0.000008070990588748828,3.848583318699639e-8,8.388415656535031e-11,6.66081165917598e-12,5.676893330419652e-11,3.922433558511962e-11,1.0048153715330743e-12,4.379984708258178e-10,5.44444038463876e-11,1.2431928316652963e-12,4.3703856505983296e-11,3.67491497302308e-12,1.1813987566000606e-9,3.3752920597374114e-11,9.566136077721055e-12,3.383305580140217e-10,1.0292429235281286e-11,9.648172885623474e-12,3.6334958508632553e-7,4.2000278255294177e-10,2.9818980618046e-10,1.2163400242570788e-7,0.000002865820988517953,1.1990112028237299e-11,8.813688112674356e-10,1.1497139351668295e-10,2.7324265072792286e-10,3.885617516630191e-8,6.343247455076195e-11,1.1135421751351515e-10,3.389848679535845e-10,0.0000013870950397176784,1.9523248795394466e-8,3.1540010186903444e-11,2.000729998341555e-10,6.31417085283914e-10,6.132015034632232e-12,0.000050130194722441956,1.7424610643246297e-9,8.75391859694874e-12,1.5106949025067706e-9,2.0486821128429256e-8,2.929064102374923e-9,1.604071186989131e-8,1.9116142002673087e-9,3.312989119153009e-11,0.000002707902012843988,7.089162156609374e-13,4.1470896161577286e-11,9.2300131981915e-9,5.905586242249683e-9,8.289174346920447e-10,7.003118224213623e-11,4.2394815435997657e-10,1.7795981355206436e-10,0.000002496192564649391,8.072198198760816e-8,1.1755798112744742e-8,1.0419590523724764e-8,1.180140998258139e-7,0.00003605161691666581,3.3121532860141967e-12,6.206528624375096e-10,3.268035286829729e-10,5.3892793006049544e-11,5.037623651560352e-9,0.000004331988293415634,3.018586047787153e-9,8.130342753209785e-12,2.0267315603561542e-11,9.819737904759318e-12,1.0164450878202835e-11,3.314166441281685e-10,3.958268166570633e-7,1.1742705696704547e-9,5.529278424010187e-12,3.5741045678605587e-9,6.243585093379522e-10,2.2291997581191936e-8,1.0736971728375622e-10,6.319770401441716e-11,0.0000021425641989480937,1.0068781745864452e-11,2.0059456873333659e-10,9.628021047092261e-9,2.0658814936513892e-11,4.180473300863241e-9,4.093985220277041e-11,2.7821872583544405e-10,9.255543380326525e-12,0.0000013490035826180247,1.971971919090265e-9,5.458582563733216e-7,2.8131779891693576e-11,2.0715922033343048e-10,1.5851441253350584e-12,3.5813123300254546e-12,1.3202635429365728e-8,8.652179667167914e-12,4.2741081074382237e-7,3.7099490146630387e-10,7.06558438423599e-8,9.648343235468815e-11,6.496301570681554e-14,4.9370008081695e-10,9.926633959267406e-10,1.4368040091028433e-10,6.297270260802179e-7,4.831603450661248e-10,1.0474292366779991e-7,1.6250416567942239e-9,0.000024217810278059915,2.30627577479936e-7,6.419081412245564e-12,1.2820641900543328e-12,4.52350903401566e-8,7.439723143098931e-11,9.71776305258576e-12,9.509596310408597e-8,1.5753748305868953e-9,1.843418126479257e-11,8.743427509783075e-11,1.1755926232481784e-9,8.699093001496294e-8,3.9812728114263507e-10,1.2599813414837513e-9,1.978183838957648e-9,3.3259859710116357e-12,1.9450257376929514e-11,4.04427713540656e-10,9.53514156520896e-10,1.1534705413041024e-9,2.884962269078528e-9,3.1039601822158147e-7,2.697875700619079e-9,7.79935938055587e-9,7.421819958608467e-8,0.000004636636731447652,2.4712067461796217e-11,4.688048560463898e-12,2.273747767844725e-8,4.744840978454512e-11,0.000003190389861629228,3.144880406438788e-12,2.712228219792223e-10,1.6377191158767346e-8,9.324459711024247e-7,2.1443204850646325e-11,2.5156179361829345e-9,1.8102656873519862e-10,4.4552702456110427e-13,8.023327835449123e-11,1.3444300783760355e-8,1.9768993733682017e-11,4.625780036793259e-10,1.6013903092471082e-8,2.5904644748719896e-11,5.744935777096671e-12,1.4746494303441438e-10,1.7657761364198166e-10,6.303413901953547e-10,3.423846206551673e-10,2.72494382613786e-9,3.0990245092343827e-12,1.9349860255757356e-12,1.2022907469599886e-7,1.2300429563794069e-9,4.291767028569149e-12,5.607136283058978e-12,5.024916038820493e-10,9.711824677793857e-10,5.52032697331839e-10,5.353419096909562e-12,2.1947594852633756e-9,1.0791332237525264e-11,3.2427923457589714e-8,7.452953185160816e-13,5.481652909367085e-10,1.4852241658758203e-9,1.1333900831744792e-11,1.3395040632246946e-10,1.0129484057097571e-10,1.0646971151939688e-11,1.7703905008659149e-9,3.1282788859332555e-12,7.005172619756195e-8,3.0423610679342072e-12,0.0000020430143194971606,2.8422849851494902e-8,4.600311884900288e-12,1.8224310949221945e-8,0.0000019336821424076334,1.7589868450684776e-9,1.9949011331732436e-9,2.2534987209610335e-8,2.9576503468575766e-9,2.7877768982165207e-9,4.3757887938089546e-12,4.138303241751906e-11,1.3785448338410333e-9,4.403904758226718e-12,1.2111565222960863e-11,3.56043244664761e-7,5.224193699859825e-8,2.0286458277118946e-11,2.267554499724156e-8,1.5387162377589192e-10,4.5694989458944235e-10,1.2388730930013025e-8,6.189968981828997e-9,4.632741672921936e-12,2.9409541468794487e-9,4.1784516818799133e-11,1.053467816453102e-11,7.148639724618988e-8,5.388615985715828e-13,6.217945269781922e-8,3.292540892707585e-11,7.705218407760839e-12,3.63891778931702e-8,1.5686217602617347e-10,1.6867884866655913e-8,9.996755778729494e-8,0.000003907777227141196,1.1768551411162065e-10,0.00006840402784291655,9.35360944254171e-9,5.244107814661447e-9,6.148508091552429e-11,1.103818103587173e-8,2.6776771446836156e-13,2.984587577081754e-11,2.1748768064266955e-10,4.713314374210631e-12,3.391163350130455e-9,7.12820913406631e-10,2.113594490538162e-7,7.450540184805732e-12,5.234562228118023e-10,5.748317932763314e-10,1.978070596209136e-9,4.221890337330336e-10,2.565163012557825e-11,1.949946497672994e-12,2.832731382795828e-9,3.05584604929493e-13,1.3135619703064094e-8,7.283212966058272e-8,1.2448271302645253e-9,3.5858010896498704e-13,6.941711649943727e-9,2.001602261714197e-8,7.164236651840961e-13,1.2701507046131155e-7,3.5962863242922083e-10,1.5264206565390737e-11,3.2076528100333235e-9,2.3620076380903576e-11,7.758771403132414e-11,3.336577758875081e-12,1.2107268312910868e-11,5.101864530843159e-7,4.3258033266546647e-10,6.286684367529105e-11,1.352925882219455e-13,3.116593205731988e-9,1.1785549966503162e-11,9.501238906750586e-8,9.142010065632533e-12,3.46550094176834e-10,9.952948465397071e-10,8.769556261722933e-13,2.4107863275113495e-11,5.238736666690613e-10,1.0780958374298488e-12,2.2885811301609493e-11,0.00002410623710602522,0.000007993623512447812,5.392549254357171e-11,4.984532964158461e-8,6.203603106769151e-7,1.1287680790061927e-9,9.435667525092839e-11,0.000001515733742962766,0.0000017070674402930308,6.312379952078118e-9,4.594496338938825e-9,2.9189681227670405e-10,9.920334952812082e-13,3.2775939740048443e-9,1.6134975300285959e-12,2.642697172205999e-10,4.6526948960234904e-11,1.4449828833473788e-10,3.2264194788078626e-11,3.627166904607293e-7,3.585314267695594e-10,2.3162957663025452e-11,1.8892793107738726e-8,0.00001149287163570989,1.464487905944445e-11,3.4905082296754486e-11,8.20483592178789e-9,0.00009694286563899368,1.1426700557981562e-10,1.8334158635724407e-9,2.4157982281280965e-8,1.2080851874429754e-8,6.470330521368695e-11,1.5025666474932642e-11,8.013452124089326e-10,1.6602545727106843e-11,6.471102764749048e-9,1.5367321637427267e-8,8.900660652955139e-9,2.1128354177335495e-8,2.1680202522134095e-8,2.350639460857451e-9,1.680395579628513e-11,0.00000486115595776937,1.4987676898170488e-12,1.9295398612229064e-11,2.0537212819249362e-8,8.181796351536264e-10,7.655917011462066e-10,0.0000674232142046094,1.6568384353377041e-7,4.351966703675103e-12,5.018244664256599e-8,9.038518197779233e-12,2.894455855551037e-11,6.784393917769194e-7,6.152335862985581e-10,5.669283185266716e-12,6.253301876313344e-9,6.616724529395768e-11,2.020877909458818e-9,5.810280642032373e-13,1.5744076042878419e-9,5.903597749545852e-11,1.01135142926978e-7,1.2164188945007481e-7,1.1042390959947745e-12,6.720439205132323e-10,1.313960919235857e-11,8.052073212638788e-7,8.358772701777539e-11,2.2157047030457733e-10,2.1157589569220647e-12,1.1424123799730346e-11,2.140300853215038e-10,9.143121815213817e-9,0.0000013994654182170052,1.9623955238579072e-10,1.2131904638876256e-12,1.165341156905697e-9,1.115385672711966e-9,4.67415395277726e-9,3.035684897900737e-11,2.4340312751447435e-11,5.257871395214497e-12,2.5201754016990208e-9,2.7368168495356393e-11,2.9249232369465972e-8,0.00010990325972670689,1.2986895114863728e-8,7.084401953960651e-10,7.759585335387342e-11,2.6377897199836298e-8,3.0726522021184977e-11,0.000002303313976881327,1.3105726226481096e-10,9.49178854625643e-7,1.6110728029428145e-10,5.051357943042234e-11,8.894335223885719e-8,0.000053102474339539185,1.249006974235467e-11,5.193704494255158e-11,1.0486269491627453e-10,1.9373698478819534e-10,2.3732242837581907e-8,3.2130076377256955e-11,1.0186789779764727e-11,5.30614385763073e-10,2.7192976403966895e-7,7.549603964207563e-8,7.392155794150312e-7,9.332723927002462e-9,7.991742684287928e-11,1.2449785646850842e-10,6.623849246878422e-10,8.26917839757968e-11,1.2671333138314367e-7,1.6254393386816446e-8,5.0056786493613e-10,4.7542719761040075e-11,1.8095158793585142e-7,3.9684749930657404e-10,1.1074124256493789e-10,1.579498416504066e-7,2.168204088492942e-10,4.394346220237821e-9,2.3591048559090666e-10,4.0389922309480575e-12,1.0280598421175124e-11,1.580301970012521e-12,4.807232016901253e-7,2.242701846455475e-8,3.2486653367413965e-9,9.792919947182455e-11,2.3497608303557627e-9,9.725052052544925e-8,6.786160078320336e-9,2.451101579481474e-7,6.6696596162652e-12,2.438939077364921e-7,1.3661032027822984e-10,2.3005060967307145e-8,1.0132256811345997e-8,5.073532739763298e-13,5.679378745071517e-9,1.2499001833532475e-8,2.761147144259013e-10,2.141625852353224e-12,2.606367162025225e-12,3.3718482228550783e-8,2.7616937359198346e-7,2.5736732744974233e-9,3.6894743260518226e-8,7.386647959961579e-10,1.8418228364680544e-7,4.10858985722129e-11,1.2950166980441047e-11,2.5173349516016685e-10,8.938283446724427e-11,2.6977586831122835e-12,8.34746161082478e-11,1.147462458384041e-9,9.116061683300813e-9,1.343394551156507e-8,4.0360799147265425e-9,4.837795095080644e-12,4.82539405766147e-7,7.79595321631632e-9,3.4332465256164957e-12,1.8496045806060124e-9,2.764184792169999e-8,0.000004120065568713471,2.903708384849324e-10,2.5650502555318866e-11,6.923011608428453e-10,5.355163423814702e-9,0.00005068378959549591,1.0378230497209984e-9,0.00003075188578804955,1.0585723270173375e-10,5.979544984158203e-12,0.0000022632834770774934,3.160204875563011e-10,3.090885533629795e-11,1.5330389735446204e-10,3.384238311809895e-8,1.1056401350018064e-9,1.1006419953218938e-7,8.430849796070561e-9,7.266893684021447e-12,3.8625754070903406e-13,1.3226082273831707e-7,1.0329466448855129e-10,1.0756445040227547e-10,3.5724827540661863e-9,2.551235915235206e-12,2.6431421495942686e-9,3.009795375170238e-13,6.39417763181882e-8,9.743844620047071e-11,1.7634073997026267e-13,2.167107292228021e-11,1.6370016453493008e-7,1.0263357004547391e-11,3.793202640878235e-9,5.807816023661871e-7,9.671780043518652e-11,1.6287428350469924e-10,1.3474400761825933e-10,8.265122630746191e-8,3.39546557537318e-10,0.000001276438979402883,2.989293534927384e-11,3.4548453507754484e-12,1.3720293347632695e-12,1.3019073319409102e-10,2.3646368196850176e-10,3.1718700582716863e-11,1.7763027715389512e-9,3.375158486029761e-10,2.2086772515084174e-12,2.700299150948382e-10,4.568399560866965e-7,2.522115778447187e-7,0.0000015563140323138214,2.486164341064523e-8,9.834174446998745e-11,2.690192513199463e-10,3.0369767500815215e-7,7.12726100360328e-9,0.00008517859532730654,7.500198684340376e-9,2.966143997085169e-11,4.5965649064783065e-9,7.376000504821789e-11,1.2932858339809172e-8,0.00002438955743855331,1.6528834123619163e-7,0.0000014316175338535686,6.286574125852207e-12,5.067156750074818e-14,3.7068595805725413e-13,3.619217991968071e-8,5.639006483981746e-10,9.056915928340459e-8,1.462930748763469e-10,8.25235821518e-13,7.520278288986049e-10,1.5911321149530977e-9,3.296793948948107e-8,1.4336819731397554e-8,7.274827140690832e-8,1.7790168643783133e-11,1.333188004437602e-10,4.749732829267828e-10,0.00027424388099461794,5.306377004465901e-10,1.8853103800342552e-10,3.131711686851779e-12,3.021751251996996e-11,2.358737205554462e-9,4.3320973475147184e-9,2.0026848235321637e-10,0.0000555994629394263,6.365068916380778e-9,1.830544604786155e-8,1.0285393647890828e-9,5.6285160560110015e-11,3.5228453043600894e-8,1.841626016130249e-7,5.215748277720422e-9,3.520469533668802e-7,1.686192629968275e-9,0.000027082855012849905,0.000003304610117993434,3.8863458939886186e-7,0.00000389463639294263,2.5827043259596394e-7,0.000015117241673578974,1.7344915193007182e-7,8.131805770972278e-7,1.346104200328213e-10,1.6013207471132773e-7,1.9429153397254595e-8,2.8232179261067358e-8,6.6226551152792634e-12,4.574805378271263e-12,5.558415949735718e-10,2.42216302570597e-10,1.97446239913468e-10,2.0345748907857342e-7,9.127479927428794e-8,2.1612727163500267e-9,2.1285720408492637e-10,0.0000013442161161947297,3.1444056247664776e-9,1.854064540784961e-10,0.000002472174855938647,7.825677605932668e-12,8.488109659587906e-10,2.856030967279821e-10,3.3829877565949573e-7,2.5352476194784312e-12,6.057065266018569e-11,2.093064166430736e-9,7.038866559928858e-13,1.3099878126698883e-10,2.2307415636912964e-10,7.244440047848855e-10,1.3959929132806304e-12,3.9929216200107476e-7,0.000006444175141950836,1.346656397505086e-10,0.00012225077080074698,5.383495818023221e-7,3.4439128171470657e-7,0.000003209426267858362,6.363997107072805e-10,4.815534992985704e-8,3.909437357174639e-11,3.680469262690167e-9,2.3815358574097445e-8,1.7666470952804048e-8,1.1924219833758798e-8,0.0003670523874461651,5.345963813851995e-7
    ];

    let type1 = {type: nn.FLOAT32};
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1000]};
    let type0_length = product(type0.dimensions);

    let input = operandIndex++;
    model.addOperand(type0);
    let beta = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type0);

    model.setOperandValue(beta, new Float32Array([1.0]));
    model.addOperation(nn.SOFTMAX, [input, beta], [output]);

    model.identifyInputsAndOutputs([input], [output]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input_input = new Float32Array(input_value);
    execution.setInput(0, input_input);

    let output_output = new Float32Array(type0_length);
    execution.setOutput(0, output_output);

    await execution.startCompute();

    for (let i = 0; i < type0_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});
