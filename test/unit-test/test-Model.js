describe('Model Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  let nn;

  beforeEach(function(){
    nn = navigator.ml.getNeuralNetworkContext();
  });

  afterEach(function(){
    nn = undefined;
  });

  describe('#addOperand API', function() {
    it('check "addOperand" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.addOperand);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel().then((model)=>{
        assert.equal(model.addOperand({type: nn.INT32}), undefined);
      });
    });

    it('passing a FLOAT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.FLOAT32});
        });
      });
    });

    it('passing an INT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.INT32});
        });
      });
    });

    it('passing an UINT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.UINT32});
        });
      });
    });

    it('passing a TENSOR_FLOAT32 tensor having "dimensions" option is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_FLOAT32 tensor not having "dimensions" option', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_FLOAT32});
        });
      });
    });

    it('passing a TENSOR_INT32 tensor having "dimensions" option is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_INT32 tensor not having "dimensions" option', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_INT32});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 0 is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 100 is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 100});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 255 is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of 0 and "zeroPoint" of [0-255] is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 100});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" < 0', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: -1});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" > 255', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 256});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of negative and "zeroPoint" of [0-255]', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, scale: -0.8, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "dimensions" options', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "scale" and "zeroPoint" options', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "scale" options', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "zeroPoint" options', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8});
        });
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand();
        });
      });
    });

    it('raise error when passing parameter is not "OperandOptions" type', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand(123);
        });
      });
    });

    it('raise error when passing two scalars', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.INT32}, {type: nn.INT32});
        });
      });
    });

    it('raise error when passing two tensors', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS},
                           {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it.skip('raise error when attempting to add an operand into the finished model', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
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
          assert.throws(() => {
            model.addOperand(op);
          });
        });
      });
    });
  });

  describe('#setOperandValue API', function() {
    it('check "setOperandValue" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.setOperandValue);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.equal(model.setOperandValue(0, new Int32Array([0])), undefined);
      });
    });

    it('raise error when setting an Int8Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('setting a Float32Array data which length equals 1 for a FLOAT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float32Array data which length > 1 for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0, 0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for a FLOAT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('setting an Int32Array data which length equals 1 for an INT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data which length > 1 for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0, 0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('raise error when setting a Float32Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for an INT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('setting an Uint32Array data which length equals 1 for an UINT32 scalar is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data which length > 1 for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0, 0]));
        });
      });
    });

    it('raise error when setting a Float32Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for an UINT32 scalar', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int8Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint8Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint8Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint8ClampedArray(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int16Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint16Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int32Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint32Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('setting a Float32Array data for a TENSOR_FLOAT32 tensor is ok', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a Float64Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Float64Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a DataView data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let buffer = new ArrayBuffer(product(op.dimensions));
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int8Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint8Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint8Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint8ClampedArray(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int16Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint16Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('setting an Int32Array data for a TENSOR_INT32 tensor is ok', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Int32Array(product(op.dimensions));
        data.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint32Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Uint32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a Float32Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a Float64Array data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let data = new Float64Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a DataView data for a TENSOR_INT32 tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        let buffer = new ArrayBuffer(product(op.dimensions));
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Int8Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('setting an Uint8Array data for a TENSOR_QUANT8_ASYMM tensor is ok', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Uint8Array(product(op.dimensions));
        data.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Uint8ClampedArray(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int16Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Int16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint16Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Uint16Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int32Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Int32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Uint32Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Uint32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a Float32Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a Float64Array data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let data = new Float64Array(product(op.dimensions));
        data.fill(0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting a DataView data for a TENSOR_QUANT8_ASYMM tensor', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let buffer = new ArrayBuffer(product(op.dimensions));
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it.skip('raise error when attempting to reset the value for an operand of the finished model', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
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
          assert.throws(() => {
            let updatData = new Float32Array(product(op.dimensions));
            data.fill(100);
            model.setOperandValue(1, data);
          });
        });
      });
    });
  });

  describe('#addOperation API', function() {
    it('check "addOperation" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.addOperation);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.equal(model.addOperation(nn.FLOOR, [0], [1]), undefined);
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperation();
        });
      });
    });

    it('raise error when passing only two parameter', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, [0]));
        });
      });
    });

    it('raise error when passing more than 3 parameter', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, [0], [1], [1]));
        });
      });
    });

    it('raise error when passing first parameter as undefined', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(undefined, [0], [1]));
        });
      });
    });

    it('raise error when passing 2nd and 3rd parameters of \'number\' type', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, 0, 1));
        });
      });
    });

    it('first two input tensors (rank<=4) of compatible dimensions having identical TENSOR_FLOAT32 type as the output tensor with third input tensor of INT32 type having value of 0-3 are ok for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand(op);
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        assert.doesNotThrow(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('first two input tensors (rank<=4) of compatible dimensions having identical TENSOR_QUANT8_ASYMM type as the output tensor with third input tensor of INT32 type having value of 0-3 are ok for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let input0Opertions = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 1, 2], scale: 0.8, zeroPoint: 0};
        model.addOperand(input0Opertions);
        let input1Opertions = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 1], scale: 0.5, zeroPoint: 1};
        model.addOperand(input1Opertions);
        let data = new Uint8Array(product(input1Opertions.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_RELU]));
        let output1pertions = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 2], scale: 0.2, zeroPoint: 10};
        model.addOperand(output1pertions);
        assert.doesNotThrow(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when one of first two input tensors doesn\'t have type of TENSOR_FLOAT32 or TENSOR_QUANT8_ASYMM for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        let input1Options = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(input1Options);
        let data = new Float32Array(product(input1Options.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when first two input tensors of compatible dimensions don\'t have identical type(TENSOR_FLOAT32 or TENSOR_QUANT8_ASYMM) for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let input0Options = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(input0Options);
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0};
        model.addOperand(op);
        let data = new Uint8Array(product(op.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when there is at least one of first two input tensors whose rank > 4 for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let TENSOR_DIMENSIONS_RANK5 = [2, 2, 2, 2, 2];
        let optionsRank5 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS_RANK5, scale: 0.8, zeroPoint: 0};
        model.addOperand(optionsRank5);
        let optionsRank4 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0};
        model.addOperand(optionsRank4);
        let data = new Uint8Array(product(optionsRank4.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(optionsRank5);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when first two input tensors whose rank are both <= 4 don\'t have compatible dimensions for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let options0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 2, 2], scale: 0.8, zeroPoint: 0};
        model.addOperand(options0);
        let options1 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 1], scale: 0.8, zeroPoint: 0};
        model.addOperand(options1);
        let data = new Uint8Array(product(options1.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 2], scale: 0.8, zeroPoint: 0});
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when first two input tensors of compatible dimensions don\'t have identical type(TENSOR_FLOAT32 or TENSOR_QUANT8_ASYMM) as the output tensor for "ADD" operation', function() {
      return nn.createModel().then((model)=>{
        let inputOptions = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(inputOptions);
        model.addOperand(inputOptions);
        let data = new Uint8Array(product(inputOptions.dimensions));
        data.fill(0);
        model.setOperandValue(1, data);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        let outputOptions = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0};
        model.addOperand(outputOptions);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3]);
        });
      });
    });

    it.skip('raise error when attempting to reset the operation of the finished model', function() {
      return nn.createModel().then((model)=>{
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
          assert.throws(() => {
            model.addOperation(nn.MUL, [0, 1, 2], [3]);
          });
        });
      });
    });
  });

  describe('#identifyInputsAndOutputs API', function() {
    it('check "identifyInputsAndOutputs" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.identifyInputsAndOutputs);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.equal(model.identifyInputsAndOutputs([0], [1]), undefined);
      });
    });

    it('raise error when letting input tensor be output tensor', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [0]);
        });
      });
    });

    it('raise error when target input(s) wasn\'t(or weren\'t) previously added', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([1], [0]);
        });
      });
    });

    it('raise error when target output(s) wasn\'t(or weren\'t) previously added', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [1]);
        });
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.identifyInputsAndOutputs();
        });
      });
    });

    it('raise error when passing only one parameter', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0]);
        });
      });
    });

    it('raise error when passing more than two parameter', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [1], [1]);
        });
      });
    });

    it('raise error when passing two parameter of \'number\' type', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs(0, 1);
        });
      });
    });

    it.skip('raise error when attempting to modify inputs/outputs of the finished model', function() {
      return nn.createModel().then((model)=>{
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
          assert.throws(() => {
            model.identifyInputsAndOutputs([3], [0]);
          });
        });
      });
    });
  });

  describe('#finish API', function() {
    it('check "finish" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.finish);
      });
    });

    it('raise error when passing a parameter', function() {
      return nn.createModel().then((model)=>{
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
        assert.throws(() => {
          model.finish(undefined);
        });
      });
    });

    it('check return value is of "Promise<long>" type after being called successfully', function() {
      return nn.createModel().then((model)=>{
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
        assert.doesNotThrow(() => {
          model.finish().then((result)=>{
            assert.strictEqual(result, 0);
          });
        });
      });
    });

    it('raise error when calling this function more than once, the function must only be called once', function() {
      return nn.createModel().then((model)=>{
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
          model.finish().catch((error)=>{
            //assert.equal(error.message, 'finish called more than once');
            assert.isOk(error);
          });
        });
      });
    });
  });

  describe('#createCompilation API', function() {
    it('check "createCompilation" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.isFunction(model.createCompilation);
      });
    });

    it('raise error when passing a parameter', function() {
      return nn.createModel().then((model)=>{
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
          assert.throws(() => {
            model.createCompilation(undefined);
          });
        });
      });
    });

    it('raise error when calling this function with model not being finished', function() {
      return nn.createModel().then((model)=>{
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
        assert.throws(() => {
          model.createCompilation();
        });
      });
    });
  });
});

