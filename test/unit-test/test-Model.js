describe('Unit Test/Model Test', function() {
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
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.addOperand);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel(options).then((model)=>{
        assert.equal(model.addOperand({type: nn.INT32}), undefined);
      });
    });

    it('passing a FLOAT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.FLOAT32});
        });
      });
    });

    it('passing an INT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.INT32});
        });
      });
    });

    it('passing an UINT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.UINT32});
        });
      });
    });

    it('passing a TENSOR_FLOAT32 tensor having "dimensions" option is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_FLOAT32 tensor not having "dimensions" option', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_FLOAT32});
        });
      });
    });

    it('passing a TENSOR_INT32 tensor having "dimensions" option is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_INT32 tensor not having "dimensions" option', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_INT32});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 0 is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 100 is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 100});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" of 255 is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of 0 and "zeroPoint" of [0-255] is ok', function() {
      return nn.createModel(options).then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 100});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" < 0', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: -1});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of non-negative and "zeroPoint" > 255', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 256});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor having "dimensions", "scale" of negative and "zeroPoint" of [0-255]', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, scale: -0.8, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "dimensions" options', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, scale: 0.8, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "scale" and "zeroPoint" options', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "scale" options', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, zeroPoint: 0});
        });
      });
    });

    it('raise error when passing a TENSOR_QUANT8_ASYMM tensor not having "zeroPoint" options', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8});
        });
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand();
        });
      });
    });

    it('raise error when passing parameter is not "OperandOptions" type', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand(123);
        });
      });
    });

    it('raise error when passing two scalars', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.INT32}, {type: nn.INT32});
        });
      });
    });

    it('raise error when passing two tensors', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS},
                           {type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        });
      });
    });

    it('raise error when attempting to add an operand into the finished model', function() {
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.setOperandValue);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.equal(model.setOperandValue(0, new Int32Array([0])), undefined);
      });
    });

    it('raise error when setting an Int8Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('setting a Float32Array data which length is 1 for a FLOAT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float32Array data which length > 1 for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0, 0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for a FLOAT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.FLOAT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('setting an Int32Array data which length is 1 for an INT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data which length > 1 for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0, 0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('raise error when setting a Float32Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for an INT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.INT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8Array([0]));
        });
      });
    });

    it('raise error when setting an Uint8ClampedArray data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint8ClampedArray([0]));
        });
      });
    });

    it('raise error when setting an Int16Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int16Array([0]));
        });
      });
    });

    it('raise error when setting an Uint16Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint16Array([0]));
        });
      });
    });

    it('raise error when setting an Int32Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Int32Array([0]));
        });
      });
    });

    it('setting an Uint32Array data which length is 1 for an UINT32 scalar is ok', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.doesNotThrow(() => {
          model.setOperandValue(0, new Uint32Array([0]));
        });
      });
    });

    it('raise error when setting an Uint32Array data which length > 1 for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Uint32Array([0, 0]));
        });
      });
    });

    it('raise error when setting a Float32Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float32Array([0]));
        });
      });
    });

    it('raise error when setting a Float64Array data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.setOperandValue(0, new Float64Array([0]));
        });
      });
    });

    it('raise error when setting a DataView data for an UINT32 scalar', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.UINT32});
        let buffer = new ArrayBuffer(1);
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when setting an Int8Array data for a TENSOR_FLOAT32 tensor', function() {
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0.8, zeroPoint: 1};
        model.addOperand(op);
        let buffer = new ArrayBuffer(product(op.dimensions));
        let data = new DataView(buffer, 0);
        assert.throws(() => {
          model.setOperandValue(0, data);
        });
      });
    });

    it('raise error when attempting to reset the value for an operand of the finished model', function() {
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.addOperation);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.equal(model.addOperation(nn.FLOOR, [0], [1]), undefined);
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.addOperation();
        });
      });
    });

    it('raise error when passing only two parameter', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, [0]));
        });
      });
    });

    it('raise error when passing more than 3 parameter', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, [0], [1], [1]));
        });
      });
    });

    it('raise error when passing first parameter as undefined', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(undefined, [0], [1]));
        });
      });
    });

    it('raise error when passing 2nd and 3rd parameters of \'number\' type', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(model.addOperation(nn.FLOOR, 0, 1));
        });
      });
    });

    it('first two input tensors (rank<=4) of compatible dimensions having identical TENSOR_FLOAT32 type as the output tensor with third input tensor of INT32 type having value of 0-3 are ok for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
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

    it('first two input tensors (rank<=4) of compatible dimensions having identical TENSOR_FLOAT32 type as the output tensor with third input tensor of INT32 type having value of 0-3 are ok for "ADD" operation/2', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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
      return nn.createModel(options).then((model)=>{
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

    it('raise error when first two input tensors whose rank are both <= 4 don\'t have compatible dimensions for "ADD" operation/2', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2]};
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand(op);
        let data = new Float32Array(product(op.dimensions));
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

    it('raise error when first two input tensors of compatible dimensions don\'t have identical type(TENSOR_FLOAT32 or TENSOR_QUANT8_ASYMM) as the output tensor for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let inputOptions = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(inputOptions);
        model.addOperand(inputOptions);
        let data = new Float32Array(product(inputOptions.dimensions));
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


    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);
        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.ADD, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);
        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.ADD, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the length of inputs is greater than 3 for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the length of inputs is less than 3 for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1], [2]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand(op);
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], [3, 4]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "ADD" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.ADD, [0, 1, 2], []);
        });
      });
    });

    it('"the length of inputs(explicit padding) being 10, 4-D tensor as input0, the type of intput1 to input8 being INT32 type, input9 also having INT32 type with value of 0-3, 4-D tensor as output" are ok for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(9, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 7, 4-D tensor as input0, the type of intput1 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output" are ok for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs is 6 (not 7 or 10) for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(5, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5], [6]);
        });
      });
    });

    it('raise error when the length of inputs is 8 (not 7 or 10) for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs is 11 (not 7 or 10) for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when input0 is of TENSOR_INT32 type (not TENSOR_FLOAT32 type or TENSOR_QUANT8_ASYMM type) for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of input0 is greater than 4 for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK5_DIMENSIONS = [100, 100, 7, 7, 3];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of input0 is less than 4 for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK3_DIMENSIONS = [100, 7, 7];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when others inputs don\'t have identical INT32 type except input0 having TENSOR_FLOAT32 type (or TENSOR_QUANT8_ASYMM type) for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "AVERAGE_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "AVERAGE_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "AVERAGE_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "AVERAGE_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the rank of output is greater than 4 tensor for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        let RANK5_DIMENSIONS = [100, 100, 7, 7, 3];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of output is less than 4 tensor for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        let RANK3_DIMENSIONS = [100, 7, 7];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throw(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7, 8]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "AVERAGE_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        assert.doesNotThrow(() => {
          model.addOperation(nn.AVERAGE_POOL_2D, [0, 1, 2, 3, 4, 5, 6], []);
        });
      });
    });

    it('"the last INT32 type scale input being 0, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0*(n-1), d1, d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being -4, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0*(n-1), d1, d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = -4;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being 1, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1*(n-1), d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 1;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being -3, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1*(n-1), d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = -3;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 4, 2, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being 2, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1, d2*(n-1), d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 2;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being -2, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1, d2*(n-1), d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = -2;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being 3, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1, d2, d3*(n-1)] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 3;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 4]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being -1, others 0 to n-1 4-D inputs having identical TENSOR_FLOAT32 type and the same dimensions [d0, d1, d2, d3], the 4-D output having same TENSOR_FLOAT32 type and [d0, d1, d2, d3*(n-1)] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = -1;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 4]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the last INT32 type scale input being 0, others 0 to n-1 4-D inputs having identical TENSOR_QUANT8_ASYMM type with the same [d0, d1, d2, d3] dimensions, the same scale and zeroPoint, the 4-D output having same TENSOR_QUANT8_ASYMM type and [d0*(n-1), d1, d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 2, 2, 2], scale: 1, zeroPoint: 1});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('"the length of inputs being 2, 4-D input0 of TENSOR_FLOAT32 type with [d0, d1, d2, d3] dimensions, input1 of INT32 type being 0, the 4-D output having same type as input0 and [d0, d1, d2, d3] dimensions" are ok for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(1, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONCATENATION, [0, 1], [2]);
        });
      });
    });

    it('raise error when 0 to n-1 4-D inputs not having identical TENSOR_FLOAT32 type but same dimensions for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when 0 to n-1 4-D inputs not having same dimensions but identical TENSOR_FLOAT32 type for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 2]});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [3, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is 4 not in (-4~3) with 0-n-1 inputs as 4-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        let axis = 4;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 4]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is -5 not in (-4~3) with 0-n-1 inputs as 4-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        let axis = -5;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is 3 not in (-3~2) with 0-n-1 inputs as 3-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]});
        model.addOperand({type: nn.INT32});
        let axis = 3;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 4]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is -4 not in (-3~2) with 0-n-1 inputs as 3-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]});
        model.addOperand({type: nn.INT32});
        let axis = -4;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is 2 not in (-2~1) with 0-n-1 inputs as 2-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
        model.addOperand({type: nn.INT32});
        let axis = 2;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 4]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is -3 not in (-2~1) with 0-n-1 inputs as 2-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
        model.addOperand({type: nn.INT32});
        let axis = -3;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is 1 not in (-1~0) with 0-n-1 inputs as 1-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2]});
        model.addOperand({type: nn.INT32});
        let axis = 1;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the last scale input is -2 not in (-1~0) with 0-n-1 inputs as 1-D tensors for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2]});
        model.addOperand({type: nn.INT32});
        let axis = -2;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when 0 to n-1 4-D inputs having identical TENSOR_QUANT8_ASYMM type with different scale for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 1, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 2, 2, 2], scale: 1, zeroPoint: 1});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when 0 to n-1 4-D inputs having identical TENSOR_QUANT8_ASYMM type with different zeroPoint for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 1});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 2, 2, 2], scale: 1, zeroPoint: 1});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the type of output is not identical with the type of 0 to n-1 inputs for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [4, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the rank of 0 to n-1 inputs is greater than 4 for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK5_DIMENSIONS = [2, 2, 2, 2, 2]
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [4, 2, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], [3, 4]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "CONCATENATION" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand(op);
        model.addOperand({type: nn.INT32});
        let axis = 0;
        model.setOperandValue(2, new Int32Array([axis]));
        assert.throws(() => {
          model.addOperation(nn.CONCATENATION, [0, 1, 2], []);
        });
      });
    });

    it('"the length of inputs (explicit padding) being 10, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input8 being INT32 type, input9 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(9, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('"the length of inputs (implicit padding) being 7, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('"the length of inputs (implicit padding) being 7, 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being equal to the product of input_scale and filter_scale, the type of input3 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output of same type as input0 and input1 with output_scale being greater than the product of input_scale and filter_scale" are ok for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.doesNotThrow(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });


    it('raise when input0 and input1 are not 4-D tensors for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when input2 is not 1-D tensor for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 7, the types of input3 to input5 are not identical INT32 type for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 10 , the types of input3 to input8 are not identical INT32 type for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) is 6 (not 7 or 10) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(5, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5], [6]);
        });
      });
    });

    it('raise error when the length of inputs is 8 (not 7 or 10) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) is 11 (not 7 or 10) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 7, the type of input6 is not INT32 type for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(6, new Float32Array([0.0]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 10, the type of input9 is not INT32 type for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(9, new Float32Array([0.0]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the type of output is not identical with the type of input0 and input1 for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the height of input0(its shape being [batches, height, width, depth_in]) is less than the filter_height of input1(its shape being [depth_out, filter_height, filter_width, depth_in]) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 33, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 1, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the width of input0(its shape being [batches, height, width, depth_in]) is less than the filter_width of input1(its shape being [depth_out, filter_height, filter_width, depth_in]) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 33, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 1, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the depth_in1 of input0(its shape being [batches, height, width, depth_in1]) is not equal to the depth_in2 of input1(its shape being [depth_out, filter_height, filter_width, depth_in2]) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the batches1 of input0(its shape being [batches1, height, width, depth_in]) is not equal to the batches2 of output(its shape being [batches2, out_height, out_width, depth_out]) for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of INT32 type having zeroPoint of 0 with bias_scale being not equal to the product of input_scale and filter_scale for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of INT32 type having bias_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 1});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being less than the product of input_scale and filter_scale for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale - 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, padingcode, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [op0, op1, op2, padingcode, stride, stride, fusecode], [op3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7, 8]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.CONV_2D, [0, 1, 2, 3, 4, 5, 6], []);
        });
      });
    });

    it('"the length of inputs(explicit padding) being 11, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input9 being INT32 type, input10 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(10, new Int32Array([depth_multiplier]));
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let depth_out = depth_in * depth_multiplier;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 8, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input6 being INT32 type, input7 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 8, 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being equal to the product of input_scale and filter_scale, the type of input3 to input6 being INT32 type, input7 also having INT32 type with value of 0-3, 4-D tensor as output of same type as input0 and input1 with output_scale being greater than the product of input_scale and filter_scale" are ok for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input_scale * filter_scale + 1.0;
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.doesNotThrow(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise when input0 and input1 are not 4-D tensors for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when input2 is not 1-D tensor for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, the types of input3 to input6 are not identical INT32 type for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, the types of input3 to input9 are not identical INT32 type for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) is 7 (not 8 or 11) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs is 9 (not 8 or 11) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(8, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8], [9]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) is 12 (not 8 or 11) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(11, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, the type of input7 is not INT32 type for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(7, new Float32Array([0.0]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, the type of input10 is not INT32 type for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(10, new Float32Array([0.0]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the type of output is not identical with the type of input0 and input1 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the height of input0(its shape being [batches, height, width, depth_in]) is less than the filter_height of input1(its shape being [1, filter_height, filter_width, depth_out]) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 33, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 1, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the width of input0(its shape being [batches, height, width, depth_in]) is less than the filter_width of input1(its shape being [1, filter_height, filter_width, depth_out]) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 33, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 1, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the batches1 of input0(its shape being [batches1, height, width, depth_in]) is not equal to the batches2 of output(its shape being [batches2, out_height, out_width, depth_out]) for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the depth of filter input1(its shape being [depth, filter_height, filter_width, depth_out]) is not 1 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being not equal to the product of input_scale and filter_scale for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having bias_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 1});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        let output_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being less than the product of input_scale and filter_scale for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input_scale * filter_scale - 0.1;
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], []);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, input0 having depth_in of shape([batches, height, width, depth_in]), and input9 specifying multiplier, the depth_out of output(its shape being [batches, out_height, out_width, depth_out]) is not equal to the product of the depth_in and the multiplier for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(6, new Int32Array([depth_multiplier]));
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let depth_out = depth_in * depth_multiplier + 1;
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, input0 having depth_in of shape([batches, height, width, depth_in]), input6 specifying multiplier, the depth_out of output(its shape being [batches, out_height, out_width, depth_out]) is not equal to the product of the depth_in and multiplier for "DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(9, new Int32Array([depth_multiplier]));
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        let depth_out = depth_in * depth_multiplier + 1;
        // Assume no padding and stride=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let stride = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, stride, stride, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('"the length of inputs (explicit padding) being 10, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input8 being INT32 type, input9 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(9, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('"the length of inputs (implicit padding) being 7, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('"the length of inputs (implicit padding) being 7, 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being equal to the product of input_scale and filter_scale, the type of input3 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output of same type as input0 and input1 with output_scale being greater than the product of input_scale and filter_scale" are ok for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise when input0 and input1 are not 4-D tensors for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when input2 is not 1-D tensor for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 7, the types of input3 to input5 are not identical INT32 type for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 10 , the types of input3 to input8 are not identical INT32 type for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) is 6 (not 7 or 10) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(5, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5], [6]);
        });
      });
    });

    it('raise error when the length of inputs is 8 (not 7 or 10) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) is 11 (not 7 or 10) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 7, the type of input6 is not INT32 type for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(6, new Float32Array([0.0]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 10, the type of input9 is not INT32 type for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(9, new Float32Array([0.0]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the type of output is not identical with the type of input0 and input1 for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the height of input0(its shape being [batches, height, width, depth_in]) is less than the filter_height of input1(its shape being [depth_out, filter_height, filter_width, depth_in]) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 33, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 1, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the width of input0(its shape being [batches, height, width, depth_in]) is less than the filter_width of input1(its shape being [depth_out, filter_height, filter_width, depth_in]) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 33, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 1, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the depth_in1 of input0(its shape being [batches, height, width, depth_in1]) is not equal to the depth_in2 of input1(its shape being [depth_out, filter_height, filter_width, depth_in2]) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the batches1 of input0(its shape being [batches1, height, width, depth_in]) is not equal to the batches2 of output(its shape being [batches2, out_height, out_width, depth_out]) for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of INT32 type having zeroPoint of 0 with bias_scale being not equal to the product of input_scale and filter_scale for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of INT32 type having bias_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 1});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being less than the product of input_scale and filter_scale for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [6, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [6], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale - 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ATROUS_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, rate, rate, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ATROUS_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, rate, rate, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ATROUS_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [op0, op1, op2, padingcode, rate, rate, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ATROUS_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [6]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [op0, op1, op2, padingcode, rate, rate, fusecode], [op3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7, 8]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "ATROUS_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_CONV_2D, [0, 1, 2, 3, 4, 5, 6], []);
        });
      });
    });

    it('"the length of inputs(explicit padding) being 11, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input9 being INT32 type, input10 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(10, new Int32Array([depth_multiplier]));
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let depth_out = depth_in * depth_multiplier;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 8, 4-D tensor as input0 of TENSOR_FLOAT32 type, 4-D tensor as input1 of TENSOR_FLOAT32 type, 1-D tensor as input2 of TENSOR_FLOAT32 type, the type of input3 to input6 being INT32 type, input7 also having INT32 type with value of 0-3, 4-D tensor as output having same type as input0 and input1" are ok for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 8, 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being equal to the product of input_scale and filter_scale, the type of input3 to input6 being INT32 type, input7 also having INT32 type with value of 0-3, 4-D tensor as output of same type as input0 and input1 with output_scale being greater than the product of input_scale and filter_scale" are ok for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input_scale * filter_scale + 1.0;
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.doesNotThrow(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise when input0 and input1 are not 4-D tensors for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when input2 is not 1-D tensor for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [6, 1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, the types of input3 to input6 are not identical INT32 type for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, the types of input3 to input9 are not identical INT32 type for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) is 7 (not 8 or 11) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs is 9 (not 8 or 11) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(8, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8], [9]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) is 12 (not 8 or 11) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(11, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, the type of input7 is not INT32 type for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(7, new Float32Array([0.0]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, the type of input10 is not INT32 type for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(10, new Float32Array([0.0]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the type of output is not identical with the type of input0 and input1 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the height of input0(its shape being [batches, height, width, depth_in]) is less than the filter_height of input1(its shape being [1, filter_height, filter_width, depth_out]) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 33, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 1, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the width of input0(its shape being [batches, height, width, depth_in]) is less than the filter_width of input1(its shape being [1, filter_height, filter_width, depth_out]) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 33, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 1, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the batches1 of input0(its shape being [batches1, height, width, depth_in]) is not equal to the batches2 of output(its shape being [batches2, out_height, out_width, depth_out]) for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the depth of filter input1(its shape being [depth, filter_height, filter_width, depth_out]) is not 1 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [101, 28, 28, 6]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having zeroPoint of 0 with bias_scale being not equal to the product of input_scale and filter_scale for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, 1-D tensor as input2 of TENSOR_INT32 type having bias_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 1});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale + 1.0;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being equal to the product of input_scale and filter_scale with zeroPoint being greater than 0 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        let output_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when 4-D tensor as input0 of TENSOR_QUANT8_ASYMM type having input_scale, 4-D tensor as input1 of TENSOR_QUANT8_ASYMM type having filter_scale, output of same type as input0 with output_scale being less than the product of input_scale and filter_scale for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 32, 32, 3], scale: input_scale, zeroPoint: 1});
        let filter_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 5, 5, 3], scale: filter_scale, zeroPoint: 2});
        let bias_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input_scale * filter_scale - 0.1;
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [100, 28, 28, 6], scale: output_scale, zeroPoint: 10});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 5]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7], []);
        });
      });
    });

    it('raise error when the length of inputs(explicit padding) being 11, input0 having depth_in of shape([batches, height, width, depth_in]), and input9 specifying multiplier, the depth_out of output(its shape being [batches, out_height, out_width, depth_out]) is not equal to the product of the depth_in and the multiplier for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(6, new Int32Array([depth_multiplier]));
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        let depth_out = depth_in * depth_multiplier + 1;
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the length of inputs(implicit padding) being 8, input0 having depth_in of shape([batches, height, width, depth_in]), input6 specifying multiplier, the depth_out of output(its shape being [batches, out_height, out_width, depth_out]) is not equal to the product of the depth_in and multiplier for "ATROUS_DEPTHWISE_CONV_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let depth_in = 3;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, depth_in]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        let depth_multiplier = 2;
        model.setOperandValue(9, new Int32Array([depth_multiplier]));
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        let depth_out = depth_in * depth_multiplier + 1;
        // Assume no padding and rate=1
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, depth_out]});
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ATROUS_DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, rate, rate, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ATROUS_DEPTHWISE_CONV_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let pad = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op0, op1, op2, pad, pad, pad, pad, rate, rate, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "ATROUS_DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, rate, rate, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "ATROUS_DEPTHWISE_CONV_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 32, 32, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 5, 5, 3]};
        let type1_length = product(type1.dimensions);
        let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [1]};
        let type2_length = product(type2.dimensions);
        let type3 = {type: nn.INT32};
        let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 28, 28, 6]};
        let type4_length = product(type4.dimensions);
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type1);
        let op2 = operandIndex++;
        model.addOperand(type2);
        let padingcode = operandIndex++;
        model.addOperand(type3);
        let rate = operandIndex++;
        model.addOperand(type3);
        let channelMultiplier = operandIndex++;
        model.addOperand(type3);
        let fusecode = operandIndex++;
        model.addOperand(type3);
        let op3 = operandIndex++;
        model.addOperand(type4);
        let op1_input = new Float32Array(type1_length);
        model.setOperandValue(op1, op1_input);
        let op2_input = new Float32Array(type2_length);
        model.setOperandValue(op2, op2_input);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(rate, new Int32Array([1]));
        model.setOperandValue(channelMultiplier, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.ATROUS_DEPTHWISE_CONV_2D, [op0, op1, op2, padingcode, rate, rate, channelMultiplier, fusecode], [op3]);
        });
      });
    });

    it('"input0 TENSOR_FLOAT32 tensor(RANK <= 4) can be converted as 2-D TENSOR_FLOAT32 tensor of shape [batch_size, input_size], input1 as 2-D TENSOR_FLOAT32 tensor of shape [num_units, input_size], input2 as 1-D TENSOR_FLOAT32 tensor of shape [num_units], input3 as INT32 scalar with value of 0-3, output TENSOR_FLOAT32 tensor of shape [batch_size, num_units]" are ok for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('"input0 TENSOR_QUANT8_ASYMM tensor(RANK <= 4) can be converted as 2-D TENSOR_QUANT8_ASYMM tensor of shape [batch_size, input_size] and its scale being \'input_scale\', input1 as 2-D TENSOR_QUANT8_ASYMM tensor of shape [num_units, input_size] and its scale being \'filter_scale\', input2 as 1-D TENSOR_INT32 tensor of shape [num_units] with zeroPoint of "0" and its scale being \'bias_scale\' which is equal to the product of \'input_scale\' and \'filter_scale\', input3 as INT32 scalar with value of 0-3, output TENSOR_QUANT8_ASYMM tensor of shape [batch_size, num_units] and its scale being \'output_scale\' which is greater than the product of \'input_scale\' and \'filter_scale\'" are ok for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.doesNotThrow(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is 1-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the rank of input0 is greater than 4 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size, 1, 1, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the type of input0 is not TENSOR_FLOAT32 or TENSOR_QUANT8_ASYMM for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input1 is 1-D tensor not 2-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input1 is 3-D tensor not 2-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_FLOAT32 tensor and input1 is not a TENSOR_FLOAT32 tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_QUANT8_ASYMM tensor and input1 is not a TENSOR_QUANT8_ASYMM tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the \'input_size_filter\' in dimensions([num_units, input_size_filter]) of input1 is not equal to the \'input_size\' in dimensions([batch_size, input_size]) of converted 2-D input0 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let input_size_filter = input_size + 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size_filter]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the \'num_units_filter\' in dimensions([num_units_filter, input_size]) of input1 is not equal to the \'num_units\' in dimensions([batch_size, num_units]) of output tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let num_units_filter = num_units + 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, num_units_filter]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input2 is 2-D tensor not 1-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, 1]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_FLOAT32 tensor and input2 is not a TENSOR_FLOAT32 tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_QUANT8_ASYMM tensor and input2 is not a TENSOR_INT32 tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the \'num_units_bias\' in dimensions([num_units_bias]) of input2 is not equal to the \'num_units\' in dimensions([batch_size, num_units]) of output tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let num_units_bias = num_units + 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, num_units]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units_bias]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the input3 for fuse code is invalid(out of 0-3) as "-1" for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([-1]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the input3 for fuse code is invalid(out of 0-3) as "4" for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([4]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the type of input3 is not INT32 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(3, new Float32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when output is 1-D tensor not 2-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when output is 3-D tensor not 2-D tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units, 1]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_FLOAT32 tensor and output is not a TENSOR_FLOAT32 tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 is a TENSOR_QUANT8_ASYMM tensor and output is not a TENSOR_QUANT8_ASYMM tensor for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the \'batch_size_output\' in dimensions([batch_size_output, num_units]) of output tensor is not equal to the \'batch_size\' in dimensions([batch_size, input_size]) of converted 2-D input0 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let batch_size_output = 4;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [batch_size_output, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 TENSOR_QUANT8_ASYMM tensor(RANK <= 4) with its scale being \'input_scale\', input1 as 2-D TENSOR_QUANT8_ASYMM tensor with its scale being \'filter_scale\', input2 as 1-D TENSOR_INT32 tensor with zeroPoint of 0 and its scale being \'bias_scale\' which isn\'t equal to the product of \'input_scale\' and \'filter_scale\' for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale + 0.1;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 TENSOR_QUANT8_ASYMM tensor(RANK <= 4) with its scale being \'input_scale\', input1 as 2-D TENSOR_QUANT8_ASYMM tensor with its scale being \'filter_scale\', input2 as 1-D TENSOR_INT32 tensor with zeroPoint being not of "0" and its scale being \'bias_scale\' which is equal to the product of \'input_scale\' and \'filter_scale\' for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let input2_zeropoint = 1;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: input2_zeropoint});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 TENSOR_QUANT8_ASYMM tensor(RANK <= 4) with its scale being \'input_scale\', input1 as 2-D TENSOR_QUANT8_ASYMM tensor with its scale being \'filter_scale\', output TENSOR_QUANT8_ASYMM tensor of shape [batch_size, num_units] and its scale being \'output_scale\' which is equal to the product of \'input_scale\' and \'filter_scale\' for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale - 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when input0 TENSOR_QUANT8_ASYMM tensor(RANK <= 4) with its scale being \'input_scale\', input1 as 2-D TENSOR_QUANT8_ASYMM tensor with its scale being \'filter_scale\', output TENSOR_QUANT8_ASYMM tensor of shape [batch_size, num_units] and its scale being \'output_scale\' which is less than the product of \'input_scale\' and \'filter_scale\' for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        let input_scale = 0.5;
        let filter_scale = 0.8;
        let bias_scale = input_scale * filter_scale;
        let output_scale = input_scale * filter_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, input_size], scale: input_scale, zeroPoint: 10});
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [num_units, input_size], scale: filter_scale, zeroPoint: 20});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [num_units], scale: bias_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [batch_size, num_units], scale: output_scale, zeroPoint: 40});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the length of inputs is less than 4 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of inputs is greater than 4 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(4, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3, 4], [5]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], []);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "FULLY_CONNECTED" operation', function() {
      return nn.createModel(options).then((model)=>{
        let batch_size = 3;
        let input_size = 1;
        let num_units = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units, input_size]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [num_units]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batch_size, num_units]});
        assert.throws(() => {
          model.addOperation(nn.FULLY_CONNECTED, [0, 1, 2, 3], [4, 5]);
        });
      });
    });

    it('"the length of inputs(explicit padding) being 10, 4-D tensor as input0, the type of intput1 to input8 being INT32 type, input9 also having INT32 type with value of 0-3, 4-D tensor as output" are ok for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(9, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);
        });
      });
    });

    it('"the length of inputs(implicit padding) being 7, 4-D tensor as input0, the type of intput1 to input5 being INT32 type, input6 also having INT32 type with value of 0-3, 4-D tensor as output" are ok for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the length of inputs is 6 (not 7 or 10) for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(5, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5], [6]);
        });
      });
    });

    it('raise error when the length of inputs is 8 (not 7 or 10) for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(7, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7], [8]);
        });
      });
    });

    it('raise error when the length of inputs is 11 (not 7 or 10) for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(10, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]);
        });
      });
    });

    it('raise error when input0 is of TENSOR_INT32 type (not TENSOR_FLOAT32 type or TENSOR_QUANT8_ASYMM type) for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of input0 is greater than 4 for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK5_DIMENSIONS = [100, 100, 7, 7, 3];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of input0 is less than 4 for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK3_DIMENSIONS = [100, 7, 7];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when others inputs don\'t have identical INT32 type except input0 having TENSOR_FLOAT32 type (or TENSOR_QUANT8_ASYMM type) for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of output is greater than 4 tensor for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        let RANK5_DIMENSIONS = [100, 100, 7, 7, 3];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the rank of output is less than 4 tensor for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(6, new Int32Array([nn.FUSED_NONE]));
        let RANK3_DIMENSIONS = [100, 7, 7];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK3_DIMENSIONS});
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [0, 1, 2, 3, 4, 5, 6], [7]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MAX_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MAX_POOL_2D" operation with explicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let pad = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(pad, new Int32Array([0]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, pad, pad, pad, pad, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MAX_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MAX_POOL_2D" operation with implicit padding', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op1 = operandIndex++;
        model.addOperand(type0);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([nn.FUSED_NONE]));
        let op2 = operandIndex++;
        model.addOperand(type0);
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], [op1, op2]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "MAX_POOL_2D" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [100, 7, 7, 3]};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let padingcode = operandIndex++;
        model.addOperand(type1);
        let stride = operandIndex++;
        model.addOperand(type1);
        let filter = operandIndex++;
        model.addOperand(type1);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        model.setOperandValue(padingcode, new Int32Array([nn.PADDING_SAME]));
        model.setOperandValue(stride, new Int32Array([1]));
        model.setOperandValue(filter, new Int32Array([2]));
        model.setOperandValue(fusecode, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.MAX_POOL_2D, [op0, padingcode, stride, stride, filter, filter, fusecode], []);
        });
      });
    });

    it('"first two input tensors (rank <= 4) of compatible dimensions having identical TENSOR_FLOAT32 type as the output tensor, input2 tensor of INT32 type having value of 0-3" are ok for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('"first two input tensors (rank <= 4) of compatible dimensions having identical TENSOR_QUANT8_ASYMM type, input2 tensor of INT32 type having value of 0-3, the output tensor also having TENSOR_QUANT8_ASYMM type with its scale is greater than the product by the scale of input0 and the scale of input1" are ok for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input0_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 1], scale: input0_scale, zeroPoint: 0});
        let input1_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 1, 2], scale: input1_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input0_scale * input1_scale + 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 2], scale: output_scale, zeroPoint: 0});
        assert.doesNotThrow(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when there is at least one of first two input tensors whose rank is greater than 4 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK5_DIMENSIONS =  [2, 2, 2, 2, 2];
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the type of input0 is different from the type of input1 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the dimension of input0 is not compatible with the dimension of input1 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 2, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the type of output is different from the type of input0 and input1 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the scale of output tensor(being of TENSOR_QUANT8_ASYMM type) is equal to the product of the input0 scale and the input1 scale for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input0_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 1], scale: input0_scale, zeroPoint: 0});
        let input1_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 1, 2], scale: input1_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input0_scale * input1_scale;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 2], scale: output_scale, zeroPoint: 0});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the scale of output tensor(being of TENSOR_QUANT8_ASYMM type) is less than the product of the input0 scale and the input1 scale for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let input0_scale = 0.5;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 1], scale: input0_scale, zeroPoint: 0});
        let input1_scale = 0.2;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [4, 1, 2], scale: input1_scale, zeroPoint: 0});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        let output_scale = input0_scale * input1_scale - 0.1;
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [5, 4, 3, 2], scale: output_scale, zeroPoint: 0});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of inputs is greater than 3 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(3, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2, 3], [4]);
        });
      });
    });

    it('raise error when the length of inputs is less than 3 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1], [2]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "4" for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);
        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);
        model.setOperandValue(fusecode, new Int32Array([4]));
        assert.throws(() => {
          model.addOperation(nn.MUL, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the value of fuse code is invalid(out of 0-3) as "-1" for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        let type0 = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        let type0_length = product(type0.dimensions);
        let type1 = {type: nn.INT32};
        let operandIndex = 0;
        let op0 = operandIndex++;
        model.addOperand(type0);
        let op1 = operandIndex++;
        model.addOperand(type0);
        let fusecode = operandIndex++;
        model.addOperand(type1);
        let op3 = operandIndex++;
        model.addOperand(type0);
        let op1_input = new Float32Array(type0_length);
        model.setOperandValue(op1, op1_input);
        model.setOperandValue(fusecode, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.MUL, [op0, op1, fusecode], [op3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 2]});
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], [3, 4]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "MUL" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [5, 4, 3, 1]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 1, 2]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([nn.FUSED_NONE]));
        assert.throws(() => {
          model.addOperation(nn.MUL, [0, 1, 2], []);
        });
      });
    });

    it('"input0 as a tensor (rank <= 4) of TENSOR_FLOAT32 type, input1 as a 1-D tensor of TENSOR_INT32 type" are ok for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 2]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 2]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.RESHAPE, [0, 1], [2]);
        });
      });
    });

    it('"input0 as a tensor (rank <= 4) of TENSOR_QUANT8_ASYMM type, input1 as a 1-D tensor of TENSOR_INT32 type" are ok for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 4], scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 2]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 2], scale: 0, zeroPoint: 0});
        assert.doesNotThrow(() => {
          model.addOperation(nn.RESHAPE, [0, 1], [2]);
        });
      });
    });

    it('raise error when the rank of input0 is greater than 4 for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        let RANK5_DIMENSIONS = [2, 2, 2, 2, 2]
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: RANK5_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([4, 8]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [4, 8]});
        assert.throws(() => {
          model.addOperation(nn.RESHAPE, [0, 1], [2]);
        });
      });
    });

    it('raise error when the number of elements in output tensor is not equal the number of elements in the input tensor for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 3]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
        assert.throws(() => {
          model.addOperation(nn.RESHAPE, [0, 1], [2]);
        });
      });
    });

    it('raise error when the length of inputs is greater than 2 for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 3]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
        assert.throws(() => {
          model.addOperation(nn.RESHAPE, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 3]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 3]});
        assert.throws(() => {
          model.addOperation(nn.RESHAPE, [0, 1], [2, 3]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "RESHAPE" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 4]});
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [2]});
        model.setOperandValue(1, new Int32Array([2, 3]));
        assert.throws(() => {
          model.addOperation(nn.RESHAPE, [0, 1], []);
        });
      });
    });

    it('"Inputs (without align_corners), input0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, height, width, depth], input1 as an INT32 scalar specifying the output tensor height(new_height > 0), input2 as an INT32 scalar specifying the output tensor width(new_width > 0), and output0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, new_height, new_width, depth]" are ok for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.doesNotThrow(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('"Inputs (with align_corners), input0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, height, width, depth], input1 as an INT32 scalar specifying the output tensor height(new_height > 0), input2 as an INT32 scalar specifying the output tensor width(new_width > 0), input3 as an INT32 scalar specifying align_corners as FALSE(value is 0), and output0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, new_height, new_width, depth]" are ok for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        let align_corners = 0;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(4, new Int32Array([align_corners]));
        assert.doesNotThrow(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('"Inputs (with align_corners), input0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, height, width, depth], input1 as an INT32 scalar specifying the output tensor height(new_height > 0), input2 as an INT32 scalar specifying the output tensor width(new_width > 0), input3 as an INT32 scalar specifying align_corners as TRUE(value is 1), and output0 as a TENSOR_FLOAT32 tensor (rank = 4) of shape [batches, new_height, new_width, depth]" are ok for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        let align_corners = 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(4, new Int32Array([align_corners]));
        assert.doesNotThrow(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('raise error when the rank of input0 is greater than 4 for "RESIZE_BILINEAR" operateion', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the rank of input0 is less than 4 for "RESIZE_BILINEAR" operateion', () => {
      return nn.createModel(options).then((model) => {
        let batches = 1;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when input0 is not TENSOR_FLOAT32 type for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when input1 is not INT32 type for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.UINT32});
        model.setOperandValue(1, new Uint32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when input2 is not INT32 type for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.UINT32});
        model.setOperandValue(2, new Uint32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when input3 of inputs (with align_corners) is not INT32 type for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.UINT32});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('raise error when input3(INT32 type) of inputs (with align_corners) has a negative value for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([-1]));
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('raise error when input3(INT32 type) of inputs (with align_corners) has a value which is greater than 1 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([2]));
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('raise error when the rank output0 is greater than 4 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the rank output0 is less than 4 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 1;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when output0 is not TENSOR_FLOAT32 type for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the output_batches of output0\'s shape([output_batches, new_height, new_width, depth]) is not equal to the input_batches of input0\'s shape([input_batches, height, width, depth]) for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let input_batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        let output_batches = input_batches - 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [input_batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [output_batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the output_height of output0\'s shape([batches, output_height, new_width, depth]) is not equal to the new_height specified by input1 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        let output_height = new_height + 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, output_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the output_width of output0\'s shape([batches, new_height, output_width, depth]) is not equal to the new_width specified by input2 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        let output_width = new_width + 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, output_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the output_depth of output0\'s shape([batches, new_height, new_width, output_depth]) is not equal to the input_depth of input0\'s shape([batches, height, width, input_depth]) for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let input_depth = 5;
        let new_height = 6;
        let new_width = 6;
        let output_depth = input_depth - 1;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, input_depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, output_depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of inputs (without align_corners) is greater than 3 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4], [3]);
        });
      });
    });

    it('raise error when the length of inputs (without align_corners) is less than 3 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1], [3]);
        });
      });
    });

    it('raise error when the length of inputs (with align_corners) is greater than 4 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2, 4, 5], [3]);
        });
      });
    });

    it('raise error when the length of inputs (with align_corners) is less than 4 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        model.addOperand({type: nn.INT32});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], [3, 4]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "RESIZE_BILINEAR" operation', () => {
      return nn.createModel(options).then((model) => {
        let batches = 2;
        let height = 3;
        let width = 4;
        let depth = 5;
        let new_height = 6;
        let new_width = 6;
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, height, width, depth]});
        model.addOperand({type: nn.INT32});
        model.setOperandValue(1, new Int32Array([new_height]));
        model.addOperand({type: nn.INT32});
        model.setOperandValue(2, new Int32Array([new_width]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [batches, new_height, new_width, depth]});
        assert.throws(() => {
          model.addOperation(nn.RESIZE_BILINEAR, [0, 1, 2], []);
        });
      });
    });

    it('"input0 as a 2-D tensor of TENSOR_FLOAT32 type, input1 as a scale of FLOAT32 type (its value being positive) and output tensor of same shape as input0" are ok for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.doesNotThrow(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('"input0 as a 4-D tensor of TENSOR_FLOAT32 type, input1 as a scale of FLOAT32 type (its value being positive) and output tensor of same shape as input0" are ok for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.doesNotThrow(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('"input0 as a 2-D tensor of TENSOR_QUANT8_ASYMM type, input1 as a scale of FLOAT32 type (its value being positive), output tensor of same shape as input0 and scale as 1.f / 256 and the zeroPoint as 0" are ok for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 4], scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 4], scale: 1.0 / 256, zeroPoint: 0});
        assert.doesNotThrow(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('"input0 as a 4-D tensor of TENSOR_QUANT8_ASYMM type, input1 as a scale of FLOAT32 type (its value being positive), output tensor of same shape as input0 and scale as 1.f / 256 and the zeroPoint as 0" are ok for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 1.0 / 256, zeroPoint: 0});
        assert.doesNotThrow(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when input1 as a scale of FLOAT32 type (its value being 0.0) for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.0]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when input1 as a scale of FLOAT32 type (its value being negative) for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([-1.0]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when output tensor is not of same shape as input0 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [1, 4]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [2, 1]});
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the scale of TENSOR_QUANT8_ASYMM type output tensor is not 1.f / 256 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 1.0, zeroPoint: 0});
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the zeroPoint of TENSOR_QUANT8_ASYMM type output tensor is not 0 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 0, zeroPoint: 0});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: TENSOR_DIMENSIONS, scale: 1.0 / 256, zeroPoint: 1});
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the rank of input0 is 1 (not 2 or 4) for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the rank of input0 is 3 (not 2 or 4) for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the rank of input0 is 5 (not 2 or 4) for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2, 2, 2, 2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2]);
        });
      });
    });

    it('raise error when the length of inputs is greater than 2 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1, 2], [3]);
        });
      });
    });

    it('raise error when the length of outputs is greater than 1 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        model.addOperand(op);
        model.addOperand(op);
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], [2, 3]);
        });
      });
    });

    it('raise error when the length of outputs is 0 not 1 for "SOFTMAX" operation', function() {
      return nn.createModel(options).then((model)=>{
        let op = {type: nn.TENSOR_FLOAT32, dimensions: [2, 2]};
        model.addOperand(op);
        model.addOperand({type: nn.FLOAT32});
        model.setOperandValue(1, new Float32Array([0.000001]));
        assert.throws(() => {
          model.addOperation(nn.SOFTMAX, [0, 1], []);
        });
      });
    });

    it('raise error when attempting to reset the operation of the finished model', function() {
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
          assert.throws(() => {
            model.addOperation(nn.MUL, [0, 1, 2], [3]);
          });
        });
      });
    });
  });

  describe('#identifyInputsAndOutputs API', function() {
    it('check "identifyInputsAndOutputs" is a function', function() {
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.identifyInputsAndOutputs);
      });
    });

    it('check return value is of "void" type', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.equal(model.identifyInputsAndOutputs([0], [1]), undefined);
      });
    });

    it('raise error when letting input tensor be output tensor', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [0]);
        });
      });
    });

    it('raise error when target input(s) wasn\'t(or weren\'t) previously added', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([1], [0]);
        });
      });
    });

    it('raise error when target output(s) wasn\'t(or weren\'t) previously added', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [1]);
        });
      });
    });

    it('raise error when passing no parameter', function() {
      return nn.createModel(options).then((model)=>{
        assert.throws(() => {
          model.identifyInputsAndOutputs();
        });
      });
    });

    it('raise error when passing only one parameter', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0]);
        });
      });
    });

    it('raise error when passing more than two parameter', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs([0], [1], [1]);
        });
      });
    });

    it('raise error when passing two parameter of \'number\' type', function() {
      return nn.createModel(options).then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS});
        assert.throws(() => {
          model.identifyInputsAndOutputs(0, 1);
        });
      });
    });

    it('raise error when attempting to modify inputs/outputs of the finished model', function() {
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
          assert.throws(() => {
            model.identifyInputsAndOutputs([3], [0]);
          });
        });
      });
    });
  });

  describe('#finish API', function() {
    it('check "finish" is a function', function() {
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.finish);
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
        assert.throws(() => {
          model.finish(undefined);
        });
      });
    });

    it('check return value is of "Promise<long>" type after being called successfully', function() {
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
        assert.doesNotThrow(() => {
          model.finish().then((result)=>{
            assert.strictEqual(result, 0);
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
      return nn.createModel(options).then((model)=>{
        assert.isFunction(model.createCompilation);
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
          assert.throws(() => {
            model.createCompilation(undefined);
          });
        });
      });
    });

    it('raise error when calling this function with model not being finished', function() {
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
        assert.throws(() => {
          model.createCompilation();
        });
      });
    });
  });
});
