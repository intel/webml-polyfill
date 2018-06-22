describe('Unit Test/Base Test', function() {
  const assert = chai.assert;

  it('check namespace', function() {
    assert(typeof navigator.ml !== 'undefined');
  });

  it('check getNeuralNetworkContext', function() {
    const nn = navigator.ml.getNeuralNetworkContext();
    assert(typeof nn !== 'undefined');
  });
});
