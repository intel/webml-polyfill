describe('End2End Test/ONNX Models', function() {
  const assert = chai.assert;
  const modelName = 'squeezenet1.1';
  const modelFile = `./end2end-test/resources/${modelName}/${modelName}.onnx`;
  let rawModel;

  before(async function () {
    this.timeout(120000);
    rawModel = await loadOnnxModel(modelFile);
  });

  for (let j=0; j<3; j++) {
    it(`check result for ${modelName} onnx model/${j+1}`, async function() {
      let inputData = await loadTensor(`./end2end-test/resources/${modelName}/test_data_set_${j}/input_0.pb`);
      let expectData = await loadTensor(`./end2end-test/resources/${modelName}/test_data_set_${j}/output_0.pb`);
      let outputData = new Float32Array(expectData.length);
      let kwargs = {
        rawModel: rawModel,
        backend: options.backend,
        prefer: options.prefer,
        softmax: false,
      };
      let model = new OnnxModelImporter(kwargs);
      await model.createCompiledModel();
      await model.compute(inputData, outputData);
      for (let i = 0; i < expectData.length; ++i) {
        assert.isTrue(almostEqualCTS(outputData[i], expectData[i]));
      }
    }).timeout(120000);
  }
});
