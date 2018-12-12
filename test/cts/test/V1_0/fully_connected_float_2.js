describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Fully connected float example/2', async function() {
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [0.503691, 0.196961, 0.521017, 0.554248, 0.288678, 0.792476, 0.561653, 0.46223, 0.650736, 0.163132, 0.029658, 0.411544, 0.470539, 0.57239, 0.538755, 0.21203];
    let op3_expect = [0, 0.0732134, 0, 0, 0, 0.280859, 0, 0.128927, 0, 0.0777251, 0, 0.270268, 0.271435, 0.0173503, 0.335465, 0.235562, 0, 0.0745866, 0, 0.051611, 0, 0.253876, 0, 0.0814873, 0, 0.104104, 0, 0.248529, 0.264194, 0, 0.302973, 0.166252];

    let type4 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [16, 8]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [16]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 16]};
    let type3_length = product(type3.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 8]};
    let type0_length = product(type0.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let b0 = operandIndex++;
    model.addOperand(type2);
    let op3 = operandIndex++;
    model.addOperand(type3);
    let act_relu = operandIndex++;
    model.addOperand(type4);

    model.setOperandValue(op2, new Float32Array([0.091327, 0.103366, -0.316505, -0.08312, 0.149366, -0.196636, -0.123672, 0.0628, 0.063031, 0.19167, -0.062001, -0.061504, -0.275581, 0.059388, -0.118497, -0.079224, 0.109758, 0.008307, -0.062657, -0.060962, -0.049782, -0.106719, -0.319482, -0.10365, 0.266455, 0.051517, -0.123448, 0.322464, 0.043282, -0.173782, -0.190381, 0.002013, 0.096086, 0.131157, 0.031164, 0.100638, -0.312191, -0.080923, -0.101318, -0.116614, 0.142238, 0.08654, -0.139154, 0.174268, -0.073161, 0.080072, 0.006874, 0.229382, -0.104321, -0.176035, -0.208587, -0.001019, -0.162032, 0.080824, -0.025021, 0.07446, -0.252595, -0.16175, -0.136403, 0.008308, 0.00571, 0.0966, 0.289839, 0.218816, -0.304651, -0.070958, 0.054598, 0.147113, -0.139112, -0.072798, -0.163335, -0.167863, -0.128762, -0.03578, 0.117262, 0.017177, 0.263335, -0.176612, 0.262961, -0.093654, -0.339283, 0.333071, 0.180827, 0.287583, 0.06635, -0.197947, -0.114449, -0.236035, 0.103532, -0.034284, 0.093299, -0.145361, 0.054001, 0.25057, 0.15701, -0.14348, -0.139061, -0.048873, 0.067557, 0.139038, 0.324106, 0.227041, 0.037793, -0.225747, -0.241619, 0.357835, 0.135762, -0.306764, -0.125982, 0.091916, 0.266587, 0.030135, 0.265148, 0.141627, 0.02012, 0.083815, -0.124556, -0.100124, -0.048159, 0.181172, 0.302309, -0.041084, 0.146334, -0.061511, -0.232605, 0.281324, 0.145408, -0.221897]));
    model.setOperandValue(b0, new Float32Array([-0.160594, 0.20577, -0.078307, -0.077984, 0.001937, 0.01586, 0.03681, 0.012346, 0.001028, 0.038551, 0.075415, 0.020804, 0.048478, -0.03227, 0.175688, -0.085662]));
    model.setOperandValue(act_relu, new Int32Array([1]));
    model.addOperation(nn.FULLY_CONNECTED, [op1, op2, b0, act_relu], [op3]);

    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);

    let op3_output = new Float32Array(type3_length);
    execution.setOutput(0, op3_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});
