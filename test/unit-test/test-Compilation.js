describe('Unit Test/Compilation Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#setPreference API', function() {
    it('check "setPreference" is a function', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then(()=>{
          model.createCompilation().then((compilation)=>{
            assert.isFunction(compilation.setPreference);
          });
        });
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            assert.equal(compilation.setPreference(nn.PREFER_LOW_POWER), undefined);
          });
        });
      });
    });

    it('passing a parameter with value being in 0-2 is ok', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            assert.doesNotThrow(() => {
              compilation.setPreference(nn.PREFER_LOW_POWER);
            });
          });
        });
      });
    });

    it('raise error when passing a parameter with value being out of 0-2', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            assert.throws(() => {
              compilation.setPreference(3);
            });
          });
        });
      });
    });

    it('raise error when passing a parameter with \'string\' type value', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            assert.throws(() => {
              compilation.setPreference('0');
            });
          });
        });
      });
    });

    it('raise error when passing two parameters with values both being in 0-2', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            assert.throws(() => {
              compilation.setPreference(nn.PREFER_LOW_POWER, nn.PREFER_LOW_POWER);
            });
          });
        });
      });
    });

    it('raise error when attempting to reset the preference of the finished compilation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            compilation.finish().then(()=>{
              assert.throws(() => {
                compilation.setPreference(nn.PREFER_LOW_POWER);
              });
            });
          });
        });
      });
    });
  });

  describe('#finish API', function() {
    it('check "finish" is a function', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then(()=>{
          model.createCompilation().then((compilation)=>{
            assert.isFunction(compilation.finish);
          });
        });
      });
    });

    it('check return value is of "Promise<long>" type', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            assert.doesNotThrow(()=>{
              compilation.finish().then((result)=>{
                console.log(result)
                assert.strictEqual(result, 0);
              });
            });
          });
        });
      });
    });

    it('raise error when passing a parameter', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            assert.throws(()=>{
              compilation.finish(undefined);
            });
          });
        });
      });
    });

    it('raise error when calling this function more than once, the function must only be called once', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then((result)=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            assert.doesNotThrow(()=>{
              compilation.finish().then(()=>{
                compilation.finish().catch((error)=>{
                  //assert.equal(error.message, 'finish called more than once');
                  assert.isOk(error);
                });
              });
            });
          });
       });
      });
    });
  });

  describe('#createExecution API', function() {
    it('check "createExecution" is a function', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then(()=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              assert.isFunction(compilation.createExecution);
            });
          });
        });
      });
    });

    it('raise error when passing a parameter', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then(()=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              assert.throws(()=>{
                compilation.createExecution(undefined);
              });
            });
          });
        });
      });
    });

    it('raise error when calling this function with compilation not being finished', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperation(nn.ADD, [0, 1, 2], [3]);
        model.identifyInputsAndOutputs([0], [3]);
        model.finish().then(()=>{
          model.createCompilation().then((compilation)=>{
            compilation.setPreference(nn.PREFER_LOW_POWER);
            assert.throws(()=>{
              compilation.createExecution();
            });
          });
        });
      });
    });
  });
});
