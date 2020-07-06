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

  it('global averagePool2d', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 3, 5, 5]});
    const output = nn.averagePool2d(input);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([
        -1.1289884 ,  0.34016284,  0.497431  ,  2.1915932 ,  0.42038894,
        -0.18261199, -0.15769927, -0.26465914,  0.03877424,  0.39492005,
        -0.33410737,  0.74918455, -1.3542547 , -0.0222946 ,  0.7094626 ,
        -0.09399617,  0.790736  , -0.75826526,  0.27656242,  0.46543223,
        -1.2342638 ,  1.1549494 ,  0.24823844,  0.75670505, -1.7108902 ,
        -1.4767597 , -1.4969662 , -0.31936142,  0.5327554 , -0.06070877,
        0.31212643,  2.2274113 ,  1.2775147 ,  0.59886885, -1.5765078 ,
        0.18522178,  0.22655599,  0.88869494,  0.38609484, -0.05860576,
        -0.72732115, -0.0046324 , -1.3593693 , -0.6295078 ,  1.384531  ,
        0.06825881,  0.19907428,  0.20298219, -0.8399954 ,  1.3583295 ,
        0.02117888, -1.0636739 , -0.30460566, -0.92678875, -0.09120782,
        -0.88333017, -0.9641269 ,  0.6065926 , -0.5830042 , -0.81138134,
        1.3569402 ,  1.2891295 ,  0.2508177 ,  0.20211531,  0.8832168 ,
        -0.19886094, -0.61088   ,  0.682026  , -0.5253442 ,  1.5022339 ,
        1.0256356 ,  1.0642492 , -0.4169051 , -0.8740329 ,  1.1494869
    ]));
    const outputBuffer = new Float32Array(3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [0.07170041, 0.05194739, 0.07117923];
    checkOutput(outputBuffer, expected);
  });
});