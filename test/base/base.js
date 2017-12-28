describe('Base Test', function() {
  const assert = chai.assert;
    
  it('check namespace', function() {
    assert(typeof navigator.ml !== 'undefined');
    assert(typeof navigator.ml.nn !== 'undefined');
  });
});