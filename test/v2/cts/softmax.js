describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('softmax', async function() {
    const input = nn.input('x', {type: 'tensor-float32', dimensions: [3, 4]});
    const output = nn.softmax(input);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('x', new Float32Array([
        0.4301911 ,  0.54719144, -1.1637765 ,  0.18390046,  0.58390397,
        0.1735679 ,  0.539724  , -0.953514  , -0.59202826, -0.17344485,
        0.14395015, -0.37920907]));
    // output shape is [3, 4]
    const outputBuffer = new Float32Array(3*4);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [0.32165375, 0.36157736, 0.0653337 , 0.25143513, 0.35271573,
        0.23400122, 0.33747196, 0.07581109, 0.17110129, 0.26004094,
        0.35717794, 0.21167983];
    checkOutput(outputBuffer, expected);
  });
});