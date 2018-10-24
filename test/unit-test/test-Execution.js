describe('Unit Test/Execution Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#setInput API', function() {
    it('check "setInput" is a function', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.isFunction(execution.setInput);
              });
            });
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
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                assert.equal(execution.setInput(0, inputData), undefined);
              });
            });
          });
        });
      });
    });

    it('raise error when the value being set to \'index\' is equal or greater than the size of inputs', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput(1, inputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when the input tensor type is \'TENSOR_FLOAT32\' and input data is not of \'Float32Arrary\' ', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Int32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput(0, inputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when the input tensor type is \'TENSOR_QUANT8_ASYMM\' and input data is not of \'Uint8Arrary\' ', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0};
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Int32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput(0, inputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing no parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setInput();
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing one parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setInput(0);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing more than two parameters', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Int32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput(0, inputData, inputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing first parameter is of \'string\' type', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Int32Array(product(op.dimensions));
                inputData.fill(1);
                assert.throws(()=>{
                  execution.setInput('0', inputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when setting invalid data to second parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setInput(0, "InvalidData");
                });
              });
            });
          });
        });
      });
    });
  });

  describe('#setOutput API', function() {
    it('check "setOutput" is a function', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.isFunction(execution.setOutput);
              });
            });
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
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Float32Array(product(op.dimensions));
                assert.equal(execution.setOutput(0, outputData), undefined);
              });
            });
          });
        });
      });
    });

    it('raise error when the value being set to \'index\' is equal or greater than the size of outputs', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Float32Array(product(op.dimensions));
                outputData.fill(1);
                assert.throws(()=>{
                  execution.setOutput(1, outputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when the output tensor type is \'TENSOR_FLOAT32\' and input data is not of \'Float32Arrary\' ', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Int32Array(product(op.dimensions));
                outputData.fill(1);
                assert.throws(()=>{
                  execution.setOutput(0, outputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when the output tensor type is \'TENSOR_QUANT8_ASYMM\' and output data is not of \'Uint8Arrary\' ', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0};
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Int32Array(product(op.dimensions));
                outputData.fill(1);
                assert.throws(()=>{
                  execution.setOutput(0, outputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing no parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setOutput();
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing one parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setOutput(0);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing more than two parameters', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Int32Array(product(op.dimensions));
                outputData.fill(1);
                assert.throws(()=>{
                  execution.setOutput(0, outputData, outputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when passing first parameter is of \'string\' type', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let outputData = new Int32Array(product(op.dimensions));
                outputData.fill(1);
                assert.throws(()=>{
                  execution.setOutput('0', outputData);
                });
              });
            });
          });
        });
      });
    });

    it('raise error when setting invalid data to second parameter', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.throws(()=>{
                  execution.setOutput(0, "InvalidData");
                });
              });
            });
          });
        });
      });
    });
  });

  describe('#startCompute API', function() {
    it('check "startCompute" is a function', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                assert.isFunction(execution.startCompute);
              });
            });
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
            compilation.setPreference(nn.PREFER_LOW_POWER);
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                execution.setInput(0, inputData);
                execution.setOutput(0, new Float32Array(product(op.dimensions)));
                assert.equal(execution.startCompute(), undefined);
              });
            });
          });
        });
      });
    });

    it('raise error when computing without inputs being ready', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                execution.setOutput(0, new Float32Array(product(op.dimensions)));
                assert.throws(()=>{
                  execution.startCompute();
                });
              });
            });
          });
        });
      });
    });

    it('raise error when computing without outputs being ready', function() {
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                execution.setInput(0, inputData);
                assert.throws(()=>{
                  execution.startCompute();
                });
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
            compilation.finish().then(()=>{
              compilation.createExecution().then((execution)=>{
                let inputData = new Float32Array(product(op.dimensions));
                inputData.fill(1);
                execution.setInput(0, inputData);
                execution.setOutput(0, new Float32Array(product(op.dimensions)));
                assert.throws(()=>{
                  execution.startCompute(undefined);
                });
              });
            });
          });
        });
      });
    });
  });
});
