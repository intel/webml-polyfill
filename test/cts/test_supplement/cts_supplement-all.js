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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 1]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 1, 2, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 2]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
    const type0 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 1, 2, 2]};
    const type1 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 1, 1]};
    const type2 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
    const length = product(type2.dimensions);

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
  
  it('check result for ATROUS_DEPTHWISE_CONV_2D example-1', async function() {
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
    console.log(op3_output);

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
});