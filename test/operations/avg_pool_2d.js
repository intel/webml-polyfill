describe('Average_pool_2d Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('check result', async function() {
    let model = await nn.createModel(options);
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    const tensorLength = product(float32TensorType.dimensions);

    model.addOperand(float32TensorType);
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});
    model.addOperand({type: nn.INT32});

    // no padding on the left
    model.setOperandValue(1, new Int32Array([0]));
    // no padding on the right
    model.setOperandValue(2, new Int32Array([0]));
    // no padding on the top
    model.setOperandValue(3, new Int32Array([0]));
    // no padding on the bottom
    model.setOperandValue(4, new Int32Array([0]));
    // set the stride as 1 when walking through input in the ‘width’ dimension
    model.setOperandValue(5, new Int32Array([1]));
    // set the stride as 1 when walking through input in the ‘height’ dimension
    model.setOperandValue(6, new Int32Array([1]));
    // set the filter width as 1
    model.setOperandValue(7, new Int32Array([1]));
    // set the filter height as 1
    model.setOperandValue(8, new Int32Array([1]));
    model.setOperandValue(9, new Int32Array([nn.FUSED_NONE]));

    model.addOperand(float32TensorType);
    model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);

    model.identifyInputsAndOutputs([0], [10]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(prefer);
    await compilation.finish();

    let execution = await compilation.createExecution();

    let inputData0 = new Float32Array(tensorLength);
    inputData0.set([1.0, 2.0, 3.0, 4.0]);
    execution.setInput(0, inputData0);

    let outputData = new Float32Array(tensorLength);
    execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], inputData0[i]));
    }
  });
});
