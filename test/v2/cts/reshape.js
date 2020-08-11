describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  async function testReshape(oldShape, newShape) {
    const input = nn.input('input', {type: 'tensor-float32', dimensions: oldShape});
    const output = nn.reshape(input, newShape);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    const bufferSize = product(oldShape);
    const inputBuffer = new Float32Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    execution.setInput('input', inputBuffer);
    const outputBuffer = new Float32Array(bufferSize);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    checkOutput(outputBuffer, inputBuffer);
  }

  it('reshape reordered_all_dims', async function() {
    testReshape([2, 3, 4], [4, 2, 3]);
  });

  it('reshape reordered_last_dims', async function() {
    testReshape([2, 3, 4], [2, 4, 3]);
  });

  it('reshape reduced_dims', async function() {
    testReshape([2, 3, 4], [2, 12]);
  });

  it('reshape extended_dims', async function() {
    testReshape([2, 3, 4], [2, 3, 2, 2]);
  });

  it('reshape one_dim', async function() {
    testReshape([2, 3, 4], [24]);
  });

  it('reshape negative_dim', async function() {
    testReshape([2, 3, 4], [2, -1, 2]);
  });

  it('reshape negative_dim', async function() {
    testReshape([2, 3, 4], [-1, 2, 3, 4]);
  });
});