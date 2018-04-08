describe('Model Test', function() {
  const assert = chai.assert;
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

    it('parameter having all options ("type", "dimensions", "scale" and "zeroPoint") is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2,2,2,2], scale: 0.8, zeroPoint: 1});
        });
      });
    });

    it('parameter having "type" and "dimensions" options is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2,2,2,2]});
        });
      });
    });

    it('parameter having "type" and "scale" options is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, scale: 0.8});
        });
      });
    });

    it('parameter having "type" and "zeroPoint" options is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, zeroPoint: 1});
        });
      });
    });

    it('parameter only having "type" option of FLOAT32 (0) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.FLOAT32});
        });
      });
    });

    it('parameter only having "type" option of INT32 (1) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.INT32});
        });
      });
    });

    it('parameter only having "type" option of UINT32 (2) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.UINT32});
        });
      });
    });

    it('parameter only having "type" option of TENSOR_FLOAT32 (3) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_FLOAT32});
        });
      });
    });

    it('parameter only having "type" option of TENSOR_INT32 (4) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_INT32});
        });
      });
    });

    it('parameter only having "type" option of TENSOR_QUANT8_ASYMM (5) is ok', function() {
      return nn.createModel().then((model)=>{
        assert.doesNotThrow(() => {
          model.addOperand({type: nn.TENSOR_QUANT8_ASYMM});
        });
      });
    });

    it('raise error when parameter having "type" option is not of above 6 types (0-5)', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: 6});
        });
      });
    });

    it('raise error when parameter doesn\'t have "type" option', function() {
      return nn.createModel().then((model)=>{
        assert.throws(model.addOperand({dimensions: [2,2,2,2], scale: 0.8, zeroPoint: 1}));
      });
    });

    it('raise error when no parameter', function() {
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

    it('raise error when passing two parameters', function() {
      return nn.createModel().then((model)=>{
        assert.throws(() => {
          model.addOperand({type: nn.INT32}, {type: nn.INT32});
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
        assert.equal(model.setOperandValue(0, new Int32Array([nn.FUSED_NONE])), undefined);
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is FLOAT32 (0), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is INT32 (1), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is UINT32 (2), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.UINT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_FLOAT32 (3), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_INT32 (4), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_INT32, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Int8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Uint8Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Uint8ClampedArray is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint8ClampedArray(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Int16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Uint16Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint16Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Int32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Int32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Uint32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Uint32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Float32Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float32Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a Float64Array is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let inputData = new Float64Array(4);
        inputData.fill(0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
        });
      });
    });

    it('target operand type is TENSOR_QUANT8_ASYMM (5), setting its value as a DataView is ok', function() {
      return nn.createModel().then((model)=>{
        model.addOperand({type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.8, zeroPoint: 1});
        let buffer = new ArrayBuffer(4);
        let inputData = new DataView(buffer, 0);
        assert.doesNotThrow(() => {
          model.setOperandValue(0, inputData);
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
