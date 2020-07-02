describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('Convolution with padding', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const filter = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
    const output = nn.conv2d(input, filter, /* padding = */ [1, 1, 1, 1]);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]));
    const outputBuffer = new Float32Array(25);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [12., 21., 27., 33., 24., 33., 54., 63., 72., 51., 63., 99., 108., 117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.];
    checkOutput(outputBuffer, expected);
  });

  it('Convolution without padding', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 5, 5]});
    const filter = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
    const output = nn.conv2d(input, filter);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]));
    const outputBuffer = new Float32Array(9);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [54., 63., 72., 99., 108., 117., 144., 153., 162.];
    checkOutput(outputBuffer, expected);
  });

  it('Convolution with strides=2 and padding', async function() {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: [1, 1, 7, 5]});
    const filter = nn.constant({type: 'tensor-float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
    const output = nn.conv2d(input, filter, /* padding = */ [1, 1, 1, 1], /* strides = */ [2, 2]);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('input', new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]));
    const outputBuffer = new Float32Array(12);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.];
    checkOutput(outputBuffer, expected);
  });
});
