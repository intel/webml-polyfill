const nn = navigator.ml.getNeuralNetworkContext();

class MobileNet {
  constructor(tfModel) {
    this._tfModel = tfModel;
    this._model = null;
    this._compilation;
    this._tensorIds = [];
    this._operandIndex = 0;
  }

  async createCompiledModel() {
    this._model = await nn.createModel();

    await this._addTensorOperands();
    await this._addOpsAndParams();

    await this._model.finish();
    this._compilation = await this._model.createCompilation();
    await this._compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    return await this._compilation.finish();
  }

  async compute(inputTensor, outputTensor) {
    let execution = await this._compilation.createExecution();

    await execution.setInput(0, inputTensor);
    await execution.setOutput(0, outputTensor);

    let error = await execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  async _addTensorOperands() {
    let graph = this._tfModel.subgraphs(0);
    let tensorsLength = graph.tensorsLength();
    for (let i = 0; i < tensorsLength; ++i) {
      let tensor = graph.tensors(i);
      let type;
      let typedArray;
      switch (tensor.type()) {
        case tflite.TensorType.FLOAT32: {
          type = nn.TENSOR_FLOAT32;
          typedArray = Float32Array;
        } break;
        case tflite.TensorType.INT32: {
          type = nn.TENSOR_INT32;
          typedArray = Int32Array;
        } break;
        default: {
          throw new Error(`tensor type ${tensor.type()} is not supproted.`);
        }
      }
      let tensorType = {type: type, dimensions: Array.from(tensor.shapeArray())};
      let tensorId = this._operandIndex++;
      await this._model.addOperand(tensorType);
      this._tensorIds.push(tensorId);
      let buffer = this._tfModel.buffers(tensor.buffer());
      if (buffer.dataLength() > 0) {
        let raw = buffer.dataArray();
        let data = new typedArray(raw.buffer, raw.byteOffset, raw.byteLength / typedArray.BYTES_PER_ELEMENT);
        await this._model.setOperandValue(tensorId, data);
      }
    }

    let inputs = Array.from(graph.inputsArray());
    let outputs = Array.from(graph.outputsArray());
    await this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  async _addScalarInt32(value) {
    const scalarInt32Type = {type: nn.INT32};
    let index = this._operandIndex++;
    await this._model.addOperand(scalarInt32Type);
    await this._model.setOperandValue(index, new Int32Array([value]));
    return index;
  }

  async _addScalarFloat32(value) {
    const scalarInt32Type = {type: nn.FLOAT32};
    let index = this._operandIndex++;
    await this._model.addOperand(scalarInt32Type);
    await this._model.setOperandValue(index, new Float32Array([value]));
    return index;
  }

  async _addOpsAndParams() {
    const PaddingCodeMap = new Map([
      [tflite.Padding.SAME, nn.PADDING_SAME],
      [tflite.Padding.VALID, nn.PADDING_VALID]
    ]);

    const FuseCodeMap = new Map([
      [tflite.ActivationFunctionType.NONE, nn.FUSED_NONE],
      [tflite.ActivationFunctionType.RELU, nn.FUSED_RELU],
      [tflite.ActivationFunctionType.RELU1, nn.FUSED_RELU1],
      [tflite.ActivationFunctionType.RELU6, nn.FUSED_RELU6],
    ]);

    let graph = this._tfModel.subgraphs(0);
    let operatorsLength = graph.operatorsLength();
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
          inputs.push(await this._addScalarInt32(paddingCode));
          inputs.push(await this._addScalarInt32(options.strideW()));
          inputs.push(await this._addScalarInt32(options.strideH()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(await this._addScalarInt32(fuseCode));
          opType = nn.CONV_2D;
        } break;
        case tflite.BuiltinOperator.DEPTHWISE_CONV_2D: {
          let options = operator.builtinOptions(new tflite.DepthwiseConv2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(await this._addScalarInt32(paddingCode));
          inputs.push(await this._addScalarInt32(options.strideW()));
          inputs.push(await this._addScalarInt32(options.strideH()));
          inputs.push(await this._addScalarInt32(options.depthMultiplier()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(await this._addScalarInt32(fuseCode));
          opType = nn.DEPTHWISE_CONV_2D;
        } break;
        case tflite.BuiltinOperator.AVERAGE_POOL_2D: {
          let options = operator.builtinOptions(new tflite.Pool2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(await this._addScalarInt32(paddingCode));
          inputs.push(await this._addScalarInt32(options.strideW()));
          inputs.push(await this._addScalarInt32(options.strideH()));
          inputs.push(await this._addScalarInt32(options.filterWidth()));
          inputs.push(await this._addScalarInt32(options.filterHeight()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(await this._addScalarInt32(fuseCode));
          opType = nn.AVERAGE_POOL_2D;
        } break;
        case tflite.BuiltinOperator.SOFTMAX: {
          let options = operator.builtinOptions(new tflite.SoftmaxOptions());
          inputs.push(await this._addScalarFloat32(options.beta()));
          opType = nn.SOFTMAX;
        } break;
        case tflite.BuiltinOperator.RESHAPE: {
          let options = operator.builtinOptions(new tflite.ReshapeOptions());
          //targetShape is in tensor
          opType = nn.RESHAPE;
        } break;
        default: {
          throw new Error(`operator type ${opcode} is not supported.`);
        }
      }
      await this._model.addOperation(opType, inputs, outputs);
    }
  }
}