class TFliteModelImporter {
  constructor(kwargs) {
    this._rawModel = kwargs.rawModel;
    this._model = null;
    this._compilation;
    this._execution;
    this._tensorIds = [];
    this._operands = [];
    this._operandIndex = 0;
    this._options = {
      softmax: kwargs.softmax,
    };
    this._backend = kwargs.backend;
    this._prefer = kwargs.prefer;
    if (this._backend === 'WebML') {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL') {
      this._nn = nnPolyfill;
    }
  }

  async createCompiledModel() {
    let options = {};
    options.backend = this._backend;
    this._model = await this._nn.createModel(options);

    this._addTensorOperands();
    this._addOpsAndParams();
    this._addInputsOutputs();

    await this._model.finish();
    this._compilation = await this._model.createCompilation();

    let start = performance.now();
    this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
    let elapsed = performance.now() - start;
    console.log(`compilation time: ${elapsed.toFixed(2)} ms`);
  }

  async compute(inputTensors, outputTensors) {
    inputTensors.forEach((inputTensor, i) => {
      this._execution.setInput(i, inputTensor);
    });
    outputTensors.forEach((outputTensor, i) => {
      this._execution.setOutput(i, outputTensor);
    });

    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  _addTensorOperands() {
    let graph = this._rawModel.subgraphs(0);
    let tensorsLength = graph.tensorsLength();
    for (let i = 0; i < tensorsLength; ++i) {
      let tensor = graph.tensors(i);
      let type;
      let typedArray;
      switch (tensor.type()) {
        case tflite.TensorType.FLOAT32: {
          type = this._nn.TENSOR_FLOAT32;
          typedArray = Float32Array;
        } break;
        case tflite.TensorType.INT32: {
          type = this._nn.TENSOR_INT32;
          typedArray = Int32Array;
        } break;
        default: {
          throw new Error(`tensor type ${tensor.type()} is not supproted.`);
        }
      }
      let dims = tensor.shapeArray().length ? Array.from(tensor.shapeArray()) : [1];
      let tensorType = {type: type, dimensions: dims};
      let tensorId = this._addOperand(tensorType);
      this._tensorIds.push(tensorId);
      let buffer = this._rawModel.buffers(tensor.buffer());
      if (buffer.dataLength() > 0) {
        let raw = buffer.dataArray();
        let data = new typedArray(raw.buffer, raw.byteOffset, raw.byteLength / typedArray.BYTES_PER_ELEMENT);
        this._setOperandValue(tensorId, data);
      }
    }
  }

  _addInputsOutputs() {
    let graph = this._rawModel.subgraphs(0);
    let inputs = Array.from(graph.inputsArray());
    let outputs = Array.from(graph.outputsArray());
    let operator = graph.operators(graph.operatorsLength()-1);
    let opCode = this._rawModel.operatorCodes(operator.opcodeIndex()).builtinCode();
    if (this._options.softmax && opCode != tflite.BuiltinOperator.SOFTMAX)
      outputs = [this._operandIndex-1];
    this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  _setOperandValue(index, value) {
    this._model.setOperandValue(index, value);
    this._operands[index] = value;
  }

  _addOperand(type, value) {
    let index = this._operandIndex++;
    this._model.addOperand(type);
    if (typeof value !== 'undefined')
      this._setOperandValue(index, value); 
    return index;
  }

  _addScalarInt32(value) {
    return this._addOperand({
      type: this._nn.INT32
    }, new Int32Array([value]));
  }

  _addScalarFloat32(value) {
    return this._addOperand({
      type: this._nn.FLOAT32
    }, new Float32Array([value]));
  }

  _addTensorFloat32(tensor, dims) {
    return this._addOperand({
      type: this._nn.TENSOR_FLOAT32,
      dimensions: dims
    }, new Float32Array(tensor));
  }

  _addOpsAndParams() {
    const PaddingCodeMap = new Map([
      [tflite.Padding.SAME, this._nn.PADDING_SAME],
      [tflite.Padding.VALID, this._nn.PADDING_VALID]
    ]);

    const FuseCodeMap = new Map([
      [tflite.ActivationFunctionType.NONE, this._nn.FUSED_NONE],
      [tflite.ActivationFunctionType.RELU, this._nn.FUSED_RELU],
      [tflite.ActivationFunctionType.RELU_N1_TO_1, this._nn.FUSED_RELU1],
      [tflite.ActivationFunctionType.RELU6, this._nn.FUSED_RELU6],
    ]);

    let graph = this._rawModel.subgraphs(0);
    let operatorsLength = graph.operatorsLength();
    for (let i = 0; i < operatorsLength; ++i) {
      let operator = graph.operators(i);
      let opCode = this._rawModel.operatorCodes(operator.opcodeIndex()).builtinCode();
      let opType;
      // some input/output tensors might be mapped to tensors
      // e.g., skipped nodes in RESIZE_BILINEAR 
      let inputs = Array.from(operator.inputsArray()).map(i => this._tensorIds[i]);
      let outputs = Array.from(operator.outputsArray()).map(i => this._tensorIds[i]);
      switch (opCode) {
        case tflite.BuiltinOperator.ADD: {
          let options = operator.builtinOptions(new tflite.AddOptions());
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
          opType = this._nn.ADD;
        } break;
        case tflite.BuiltinOperator.CONV_2D: {
          let options = operator.builtinOptions(new tflite.Conv2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(paddingCode));
          if (options.dilationWFactor() !== 1 || options.dilationWFactor() !== 1) {
            inputs.push(this._addScalarInt32(options.dilationWFactor()));
            inputs.push(this._addScalarInt32(options.dilationHFactor()));
            opType = this._nn.ATROUS_CONV_2D;
          } else {
            inputs.push(this._addScalarInt32(options.strideW()));
            inputs.push(this._addScalarInt32(options.strideH()));
            opType = this._nn.CONV_2D;
          }
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
        } break;
        case tflite.BuiltinOperator.DEPTHWISE_CONV_2D: {
          let options = operator.builtinOptions(new tflite.DepthwiseConv2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(paddingCode));
          if (options.dilationWFactor() !== 1 || options.dilationWFactor() !== 1) {
            inputs.push(this._addScalarInt32(options.dilationWFactor()));
            inputs.push(this._addScalarInt32(options.dilationHFactor()));
            opType = this._nn.ATROUS_DEPTHWISE_CONV_2D;
          } else {
            inputs.push(this._addScalarInt32(options.strideW()));
            inputs.push(this._addScalarInt32(options.strideH()));
            opType = this._nn.DEPTHWISE_CONV_2D;
          }
          inputs.push(this._addScalarInt32(options.depthMultiplier()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
        } break;
        case tflite.BuiltinOperator.AVERAGE_POOL_2D: {
          let options = operator.builtinOptions(new tflite.Pool2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(paddingCode));
          inputs.push(this._addScalarInt32(options.strideW()));
          inputs.push(this._addScalarInt32(options.strideH()));
          inputs.push(this._addScalarInt32(options.filterWidth()));
          inputs.push(this._addScalarInt32(options.filterHeight()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
          opType = this._nn.AVERAGE_POOL_2D;
        } break;
        case tflite.BuiltinOperator.MAX_POOL_2D: {
          let options = operator.builtinOptions(new tflite.Pool2DOptions());
          let paddingCode = PaddingCodeMap.get(options.padding());
          if (typeof paddingCode === 'undefined') {
            throw new Error(`Padding code ${options.padding()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(paddingCode));
          inputs.push(this._addScalarInt32(options.strideW()));
          inputs.push(this._addScalarInt32(options.strideH()));
          inputs.push(this._addScalarInt32(options.filterWidth()));
          inputs.push(this._addScalarInt32(options.filterHeight()));
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
          opType = this._nn.MAX_POOL_2D;
        } break;
        case tflite.BuiltinOperator.SOFTMAX: {
          let options = operator.builtinOptions(new tflite.SoftmaxOptions());
          inputs.push(this._addScalarFloat32(options.beta()));
          opType = this._nn.SOFTMAX;
        } break;
        case tflite.BuiltinOperator.CONCATENATION: {
          let options = operator.builtinOptions(new tflite.ConcatenationOptions());
          inputs.push(this._addScalarInt32(options.axis()));
          opType = this._nn.CONCATENATION;
        } break;
        case tflite.BuiltinOperator.RESHAPE: {
          let options = operator.builtinOptions(new tflite.ReshapeOptions());
          // targetShape is in tensor
          opType = this._nn.RESHAPE;
        } break;
        case tflite.BuiltinOperator.SQUEEZE: {
          let options = operator.builtinOptions(new tflite.SqueezeOptions());
          let tensorType = {type: this._nn.TENSOR_INT32, dimensions: [2]};
          let tensorId = this._operandIndex++;
          this._model.addOperand(tensorType);
          this._tensorIds.push(tensorId);
          this._model.setOperandValue(tensorId, new Int32Array([1, 1001]));
          inputs.push(tensorId);
          opType = this._nn.RESHAPE;         
        } break;
        case tflite.BuiltinOperator.FULLY_CONNECTED: {
          let options = operator.builtinOptions(new tflite.FullyConnectedOptions());
          let fuseCode = FuseCodeMap.get(options.fusedActivationFunction());
          if (typeof fuseCode === 'undefined') {
            throw new Error(`Fuse code ${options.fusedActivationFunction()} is not supported.`);
          }
          inputs.push(this._addScalarInt32(fuseCode));
          opType = this._nn.FULLY_CONNECTED;
        } break;
        case tflite.BuiltinOperator.RESIZE_BILINEAR: {

          let newSize = this._operands[inputs[1]];
          let oldSize = graph.tensors(inputs[0]).shapeArray().slice(1, 3);
          if (newSize[0] === oldSize[0] && newSize[1] === oldSize[1]) {
            // skip RESIZE_BILINEAR with the same input and output shape
            this._tensorIds[outputs[0]] = this._tensorIds[inputs[0]];
            continue;
          }

          inputs = [inputs[0]];
          inputs.push(this._addScalarInt32(newSize[0]));
          inputs.push(this._addScalarInt32(newSize[1]));

          opType = this._nn.RESIZE_BILINEAR;
        } break;
        default: {
          throw new Error(`operator type ${opCode} is not supported.`);
        }
      }

      if (i === operatorsLength - 1) { 
        if (this._options.softmax && opCode != tflite.BuiltinOperator.SOFTMAX) {
          this._model.addOperation(opType, inputs, outputs);
          let outputTensor = graph.tensors(outputs[0]);
          // Add inputs
          inputs = [];
          inputs.push(outputs[0]);
          // Set beta to 1.0
          inputs.push(this._addScalarFloat32(1.0));
          // Add outputs
          outputs = [];
          let tensorType = {type: this._nn.TENSOR_FLOAT32, dimensions: Array.from(outputTensor.shapeArray())};
          let tensorId = this._addOperand(tensorType);
          this._tensorIds.push(tensorId);
          outputs.push(tensorId);

          opType = this._nn.SOFTMAX;
        }
      }

      this._model.addOperation(opType, inputs, outputs);
    }
  }
}