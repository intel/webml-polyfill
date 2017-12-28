describe('Base Test', function() {
  const assert = chai.assert;
    
  it('check namespace', function() {
    assert(typeof navigator.ml !== 'undefined');
    assert(typeof navigator.ml.nn !== 'undefined');
    assert(typeof navigator.ml.nn.Compilation !== 'undefined');
    assert(typeof navigator.ml.nn.Execution !== 'undefined');
    assert(typeof navigator.ml.nn.Memory !== 'undefined');
    assert(typeof navigator.ml.nn.Model !== 'undefined');
  });
});