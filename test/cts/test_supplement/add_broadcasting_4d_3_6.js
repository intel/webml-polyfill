describe('CTS Supplement Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

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
});
