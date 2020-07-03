describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('maxPool2d', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 4, 4]});
    const windowDimensions = [3, 3];
    const output = nn.maxPool2d(input, windowDimensions);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    const outputBuffer = new Float32Array(4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [11, 12, 15, 16];
    checkOutput(outputBuffer, expected);
  });

  it('maxPool2d dilations', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 4, 4]});
    const windowDimensions = [2, 2];
    const dilations = [2, 2];
    const output = nn.maxPool2d(input, windowDimensions, undefined, undefined, dilations);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    const outputBuffer = new Float32Array(4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [11, 12, 15, 16];
    checkOutput(outputBuffer, expected);
  });

  it('maxPool2d pads', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const output = nn.maxPool2d(input, windowDimensions, padding);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]));
    const outputBuffer = new Float32Array(25);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25];
    checkOutput(outputBuffer, expected);
  });

  it('maxPool2d strides', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const output = nn.maxPool2d(input, windowDimensions, undefined, strides);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]));
    const outputBuffer = new Float32Array(4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [7, 9, 17, 19];
    checkOutput(outputBuffer, expected);
  });

  it('averagePool2d', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 4, 4]});
    const windowDimensions = [3, 3];
    const output = nn.averagePool2d(input, windowDimensions);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    const outputBuffer = new Float32Array(4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [6, 7, 10, 11];
    checkOutput(outputBuffer, expected);
  });

  it('averagePool2d pads', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const windowDimensions = [5, 5];
    const padding = [2, 2, 2, 2];
    const output = nn.averagePool2d(input, windowDimensions, padding);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]));
    const outputBuffer = new Float32Array(25);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19];
    checkOutput(outputBuffer, expected);
  });

  it('averagePool2d strides', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const windowDimensions = [2, 2];
    const strides = [2, 2];
    const output = nn.averagePool2d(input, windowDimensions, undefined, strides);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]));
    const outputBuffer = new Float32Array(4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [4, 6, 14, 16];
    checkOutput(outputBuffer, expected);
  });
});