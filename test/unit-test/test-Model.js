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
        assert.equal(typeof model.addOperand, 'function');
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
  });

  describe('#setOperandValue API', function() {
    it('check "setOperandValue" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.equal(typeof model.setOperandValue, 'function');
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
  });

  describe('#addOperation API', function() {
    it('check "addOperation" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.equal(typeof model.addOperation, 'function');
      });
    });
  });

  describe('#identifyInputsAndOutputs API', function() {
    it('check "identifyInputsAndOutputs" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.equal(typeof model.identifyInputsAndOutputs, 'function');
      });
    });
  });

  describe('#finish API', function() {
    it('check "finish" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.equal(typeof model.finish, 'function');
      });
    });
  });

  describe('#createCompilation API', function() {
    it('check "createCompilation" is a function', function() {
      return nn.createModel().then((model)=>{
        assert.equal(typeof model.createCompilation, 'function');
      });
    });
  });
});
