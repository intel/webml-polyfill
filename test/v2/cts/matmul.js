describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext('v2');

  it('matmul 1d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [4]},
        new Float32Array([0.8782074 , 0.22533207, 0.7134056 , 0.04190519]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([0.9025404 , 0.89538723, 0.16789329, 0.7440875]));
    // output shape is scalar
    const outputBuffer = new Float32Array(1);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [1.1453342];
    checkOutput(outputBuffer, expected);
  });

  it('matmul 1dx2d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [4, 3]},
        new Float32Array([
            0.3093976 , -1.2924036 , -0.64339244,  1.1423386 ,  1.5052135 ,
            1.8182521 , -1.825652  , -0.39694095, -0.90111053,  0.7807154 ,
            -1.9163561 , -0.13988003
        ]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([0.1309212 , 0.9090703 , 0.62183434, 0.9195683]));
    // output shape is [3]
    const outputBuffer = new Float32Array(3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [0.6616409 , -0.80990994,  0.8797145];
    checkOutput(outputBuffer, expected);
  });

  it('matmul 2dx1d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [3, 4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [4]},
        new Float32Array([0.25528687, 0.2126722 , 0.26320502, 0.8297401]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([
        0.3582649 , 0.83665735, 0.30253866, 0.6446781 , 0.4684662 ,
        0.94761264, 0.4122941 , 0.6787481 , 0.15072346, 0.2820577 ,
        0.67296237, 0.3856028
    ]));
    // output shape is [3]
    const outputBuffer = new Float32Array(3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [0.8839391, 0.9928265, 0.5955407];
    checkOutput(outputBuffer, expected);
  });

  it('matmul 2d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [3, 4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [4, 3]},
        new Float32Array([
            0.17467105, -1.2045133 , -0.02621938,  0.6096196 ,  1.4499376 ,
            1.3465316 ,  0.03289436,  1.0754977 , -0.61485314,  0.94857556,
            -0.36462623,  1.402278]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([
        0.9602246 ,  0.97682184, -0.33201018,  0.8248904 ,  0.40872088,
        0.18995902,  0.69355214, -0.37210146,  0.18104352,  3.270753  ,
        -0.803097  , -0.7268995]));
    // output shape is [3, 3]
    const outputBuffer = new Float32Array(3*3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [1.5347629 , -0.3981255 ,  2.6510081 , -0.14295794,  0.6647107 , -0.70315295,  1.3096018 ,  3.9256358 ,  3.873897];
    checkOutput(outputBuffer, expected);
  });

  it('matmul 3d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [2, 3, 4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [2, 4, 3]},
        new Float32Array([
            -2.7142005 ,  0.41909233,  0.80572236,  0.19983047, -1.9361104 ,
            1.1919757 ,  0.61684674,  0.23732206,  0.74679494,  0.4595843 ,
            -0.90667343,  0.7676448 ,  0.48643762,  0.41120672,  1.1319419 ,
            1.9692143 , -0.44463134,  0.17005378,  1.1589569 , -0.4333597 ,
            -0.47976026,  0.01067371, -0.79455626, -1.4024538]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([
        0.19521078,  0.11637875,  0.54684865,  0.13257395, -0.05654722,
        -0.64351636, -1.0019655 , -1.6156989 ,  0.01625126,  1.2386297 ,
        -0.1242797 ,  0.40350053, -0.5883816 ,  0.93452644, -0.01409106,
        -0.7825521 , -1.2281458 , -1.2388189 ,  0.7644939 , -0.8567167 ,
        0.3942727 , -0.772506  , -0.06412488, -0.9848109]));
    // output shape is [2, 3, 3]
    const outputBuffer = new Float32Array(2*3*3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [
        -0.10833447, -0.13393278,  0.8061598 , -1.3357227 ,  2.449343  ,
        -2.801163  ,  0.31218773, -2.7866507 ,  1.7064441 ,  1.5293882 ,
        -0.02957799,  0.5971595 , -2.1600451 ,  0.39520463, -0.7661238 ,
        -1.4142704 ,  1.3158847 ,  1.7268425
    ];
    checkOutput(outputBuffer, expected);
  });

  it('matmul 4d', async function() {
    const a = nn.input('a', {type: 'tensor-float32', dimensions: [1, 2, 3, 4]});
    const b = nn.constant({type: 'tensor-float32', dimensions: [1, 2, 4, 3]},
        new Float32Array([
            -0.45605758, -0.43318668,  0.61509126, -2.2228749 ,  0.50257015,
            -0.29311436, -0.64561933, -0.6439757 ,  1.6211574 , -0.28852704,
            -0.46247238,  0.5082442 ,  1.2357981 , -0.82043344, -0.926581  ,
            -0.8955289 ,  0.74586314, -0.8022598 , -0.5360306 , -0.08719682,
            0.72717273,  1.1277325 ,  2.0261378 , -1.4311641]));
    const output = nn.matmul(a, b);
    const model = await nn.createModel([{name: 'output', operand: output}]);
    const compilation = await model.createCompilation();
    const execution = await compilation.createExecution();
    execution.setInput('a', new Float32Array([
        -0.8074054 , -0.72524256,  0.4510249 ,  1.6203358 ,  1.9851393 ,
        0.501528  ,  1.3975041 , -2.3231244 ,  0.70866925,  0.24667543,
        -0.6271161 , -0.9634111 , -0.5911732 , -0.09888726, -1.0926677 ,
        0.47262478,  0.6141726 , -0.634484  , -0.07425678, -1.2638812 ,
        -1.1002079 , -1.5324054 , -1.1643038 , -0.05644368]));
    // output shape is [1, 2, 3, 3]
    const outputBuffer = new Float32Array(2*3*3);
    execution.setOutput('output', outputBuffer);
    await execution.startCompute();
    const expected = [
        1.2216457 , -1.0545375 ,  1.2706597 , -2.2521434 , -0.4334606 ,
        2.1588962 , -0.1886742 ,  0.66638416, -1.1427099 ,  0.47668338,
        1.464142  , -0.84385866, -0.058324  , -3.5314486 ,  1.6947643 ,
        0.5731275 , -0.2531564 ,  1.4829493
    ];
    checkOutput(outputBuffer, expected);
  });
});