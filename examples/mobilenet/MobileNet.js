const nn = navigator.ml.nn;

class MobileNet {
  constructor(tfModel) {
    this._tfModel = tfModel;
    this._model = null;
    this._compilation;
    this._tensorIds = [];
  }

  async createCompiledModel() {
    this._model = new nn.Model('MobileNet');

    this._addTensorOperands();
    this._addOpsAndParams();

    this._model.finish();
    this._compilation = new nn.Compilation(this._model);
    this._compilation.setPreference(nn.PreferenceCode.FAST_SINGLE_ANSWER);
    return await this._compilation.finish();
  }

  async compute(input) {
  }

  _addTensorOperands() {
    let graph = this._tfModel.subgraphs(0);
    let tensorsLength = graph.tensorsLength();
    for (let i = 0; i < tensorsLength; ++i) {
      let tensor = graph.tensors(i);
      let type;
      let typedArray;
      switch (tensor.type()) {
        case tflite.TensorType.FLOAT32: {
          type = nn.OperandCode.TENSOR_FLOAT32;
          typedArray = Float32Array;
        } break;
        case tflite.TensorType.INT32: {
          type = nn.OperandCode.TENSOR_INT32;
          typedArray = Int32Array;
        } break;
        default: {
          throw new Error(`tensor type ${tensor.type()} is not supproted.`);
        }
      }
      let tensorType = {type: type, dimensions: Array.from(tensor.shapeArray())};
      let tensorId = this._model.addOperand(tensorType);
      this._tensorIds.push(tensorId);
      let buffer = this._tfModel.buffers(tensor.buffer());
      if (buffer.dataLength() > 0) {
        let raw = buffer.dataArray();
        let data = new typedArray(raw.buffer, raw.byteOffset, raw.byteLength / typedArray.BYTES_PER_ELEMENT);
        this._model.setOperandValue(tensorId, data);
      }
    }
  }

  _addOpsAndParams() {
    let graph = this._tfModel.subgraphs(0);
    let operatorsLength = graph.operatorsLength();
    let model = this._model;

    function addScalarInt32(inputs, value) {
      const scalarInt32Type = {type: nn.OperandCode.INT32};
      let index = model.addOperand(scalarInt32Type);
      model.setOperandValue(index, value);
      inputs.push(index);
    }

    function addScalarFloat32(inputs, value) {
      const scalarInt32Type = {type: nn.OperandCode.FLOAT32};
      let index = model.addOperand(scalarInt32Type);
      model.setOperandValue(index, value);
      inputs.push(index);
    }

    const PaddingCodeMap = new Map([
      [tflite.Padding.SAME, nn.PaddingCode.SAME],
      [tflite.Padding.VALID, nn.PaddingCode.VALID]
    ]);

    const FuseCodeMap = new Map([
      [tflite.ActivationFunctionType.NONE, nn.FuseCode.NONE],
      [tflite.ActivationFunctionType.RELU, nn.FuseCode.RELU],
      [tflite.ActivationFunctionType.RELU1, nn.FuseCode.RELU1],
      [tflite.ActivationFunctionType.RELU6, nn.FuseCode.RELU6],
    ]);

    for (let i = 0; i < operatorsLength; ++i) {
      let operator = graph.operators(i);
      let opCode = this._tfModel.operatorCodes(operator.opcodeIndex()).builtinCode();
      let opType;
      let inputs = Array.from(operator.inputsArray());
      let outputs = Array.from(operator.outputsArray());
      switch (opCode) {
        case tflite.BuiltinOperator.CONV_2D: {
          let options = operator.builtinOptions(new tflite.Conv2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          addScalarInt32(inputs, paddingCode);
          addScalarInt32(inputs, options.strideW());
          addScalarInt32(inputs, options.strideH());
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          addScalarInt32(inputs, fuseCode);
          opType = nn.OperationCode.CONV_2D;
        } break;
        case tflite.BuiltinOperator.DEPTHWISE_CONV_2D: {
          let options = operator.builtinOptions(new tflite.DepthwiseConv2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          addScalarInt32(inputs, paddingCode);
          addScalarInt32(inputs, options.strideW());
          addScalarInt32(inputs, options.strideH());
          addScalarInt32(inputs, options.depthMultiplier());
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          addScalarInt32(inputs, fuseCode);
          opType = nn.OperationCode.DEPTHWISE_CONV_2D;
        } break;
        case tflite.BuiltinOperator.AVERAGE_POOL_2D: {
          let options = operator.builtinOptions(new tflite.Pool2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          addScalarInt32(inputs, paddingCode);
          addScalarInt32(inputs, options.strideW());
          addScalarInt32(inputs, options.strideH());
          addScalarInt32(inputs, options.filterWidth());
          addScalarInt32(inputs, options.filterHeight());
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          addScalarInt32(inputs, fuseCode);
          opType = nn.OperationCode.AVERAGE_POOL_2D;
        } break;
        case tflite.BuiltinOperator.SOFTMAX: {
          let options = operator.builtinOptions(new tflite.SoftmaxOptions());
          addScalarFloat32(inputs, options.beta());
          opType = nn.OperationCode.SOFTMAX;
        } break;
        case tflite.BuiltinOperator.RESHAPE: {
          let options = operator.builtinOptions(new tflite.ReshapeOptions());
          //targetShape is in tensor
          opType = nn.OperationCode.RESHAPE;
        } break;
        default: {
          throw new Error(`operator type ${opcode} is not supported.`);
        }
      }
      model.addOperation(opType, inputs, outputs);
    }
  }
}