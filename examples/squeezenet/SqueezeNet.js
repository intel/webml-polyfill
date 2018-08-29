class SqueezeNet {
  constructor(onnxModel, backend) {
    this._onnxModel = onnxModel;
    this._model = null;
    this._compilation;
    this._execution;
    this._tensorIds = [];
    this._operandIndex = 0;
    this._urlPrefer = getPreferParam();
    if (typeof backend !== 'undefined') {
      this._backend = backend;
      if (getOS() === 'Mac OS' && backend === 'WebML' && this._urlPrefer === 'invalid') {
        this._backend = 'WASM';
      }
    } else {
      if (nnNative && this._urlPrefer !== 'invalid') {
        this._backend = 'WebML';
      } else {
        this._backend = 'WASM';
      }
    }
    if (nativeBackendArray.indexOf(this._backend) !== -1) {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL2') {
      this._nn = nnPolyfill;
    }
  }

  async createCompiledModel() {
    let options = {};
    if (this._backend === 'WebGL2') {
      options.useWebGL2 = true;
    }
    this._model = await this._nn.createModel(options);

    this._addTensorOperands();
    this._addOpsAndParams();

    await this._model.finish();
    this._compilation = await this._model.createCompilation();
    this._compilation.setPreference(this._getPrefer());
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
  }

  async compute(inputTensor, outputTensor) {
    this._execution.setInput(0, inputTensor);
    this._execution.setOutput(0, outputTensor);

    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  _getPrefer() {
    let prefer = this._nn.PREFER_FAST_SINGLE_ANSWER;
    if (getOS() === 'Mac OS') {
      if (this._backend === 'MPS') {
        prefer = this._nn.PREFER_SUSTAINED_SPEED;
      } else if (this._backend === 'WebML') {
        if (this._urlPrefer === 'sustained') {
          prefer = this._nn.PREFER_SUSTAINED_SPEED;
        } else if (this._urlPrefer === 'fast') {
          prefer = this._nn.PREFER_FAST_SINGLE_ANSWER;
        }
      }
    }
    return prefer;
  }

  _getInputByName(name) {
    return getObjectByName(this._onnxModel.graph.input, name);
  }

  _getOutputByName(name) {
    return getObjectByName(this._onnxModel.graph.output, name);
  }

  _getInitializerByName(name) {
    return getObjectByName(this._onnxModel.graph.initializer, name);
  }

  _addTensorOperands() {
    const graph = this._onnxModel.graph;
    for (let i = 0; i < graph.input.length; ++i) {
      this._addTensorByValueInfo(graph.input[i]);
    }
    for (let i = 0; i < graph.output.length; ++i) {
      this._addTensorByValueInfo(graph.output[i]);
    }
    let inputs = [this._getTensorIdByName(graph.input[graph.input.length-1].name)];
    let outputs = [this._getTensorIdByName(graph.output[0].name)];
    this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  _addTensorByValueInfo(valueInfo) {
    const name = valueInfo.name;
    if (this._tensorIds[name])
      throw new Error(`Tensor ${name} is already added`);
    let tensorType = valueInfo.type.tensorType;
    let dims = tensorType.shape.dim.map(dim => {return dim.dimValue;});
    if (dims.length == 4) {
      // NCHW -> NHWC
      let nchw = Array.from(dims);
      dims[0] = nchw[0];
      dims[1] = nchw[2];
      dims[2] = nchw[3];
      dims[3] = nchw[1];
    }
    
    let type;
    switch (tensorType.elemType) {
      case onnx.TensorProto.DataType.FLOAT: {
        type = this._nn.TENSOR_FLOAT32;
      } break;
      case onnx.TensorProto.DataType.FLOAT: {
        type = this._nn.TENSOR_INT32;
      } break;
      default: {
        throw new Error(`tensor type ${tensorType.elemType} is not supproted.`);
      }
    }
    const operandType = {type: type, dimensions: Array.from(dims)};
    const tensorId = this._operandIndex++;
    this._tensorIds[name] = {id: tensorId, type: operandType};
    this._model.addOperand(operandType);

    // set operand value
    const initializer = getObjectByName(this._onnxModel.graph.initializer, name);
    if (initializer) {
      let data;
      if (initializer.dataType == onnx.TensorProto.DataType.FLOAT) {
        if (initializer.floatData && initializer.floatData.length > 0) {
          data = initializer.floatData;
        } else if (initializer.rawData && initializer.rawData.length > 0) {
          let dataView = new DataView(initializer.rawData.buffer, initializer.rawData.byteOffset, initializer.rawData.byteLength);
          let length = product(initializer.dims);
          data = new Float32Array(length);
          for (let i = 0; i < length; ++i) {
            // raw data is stored in little-endian order
            data[i] = dataView.getFloat32(i*Float32Array.BYTES_PER_ELEMENT, true);
          }
        }
      }
      if (initializer.dims.length === 4) {
        // NCHW -> NHWC
        let nhwcData = new Float32Array(data.length);
        const N = initializer.dims[0];
        const C = initializer.dims[1];
        const H = initializer.dims[2];
        const W = initializer.dims[3];
        for (let n = 0; n < N; ++n) {
          for (let c = 0; c < C; ++c) {
            for (let h = 0; h < H; ++h) {
              for (let w = 0; w < W; ++w) {
                nhwcData[n*H*W*C + h*W*C + w*C + c] = data[n*C*H*W + c*H*W + h*W + w];
              }
            }
          }
        }
        data = nhwcData;
      }
      this._model.setOperandValue(tensorId, data);
      console.log(`set operand ${name} data ${data.length}`);
    }
    return tensorId;
  }

  _addScalarInt32(value) {
    const scalarInt32Type = {type: this._nn.INT32};
    let index = this._operandIndex++;
    this._model.addOperand(scalarInt32Type);
    this._model.setOperandValue(index, new Int32Array([value]));
    return index;
  }

  _addScalarFloat32(value) {
    const scalarInt32Type = {type: this._nn.FLOAT32};
    let index = this._operandIndex++;
    this._model.addOperand(scalarInt32Type);
    this._model.setOperandValue(index, new Float32Array([value]));
    return index;
  }

  _getTensorIdByName(name) {
    let info = this._tensorIds[name];
    if (typeof info === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return info.id;
  }

  _getTensorTypeByName(name) {
    let info = this._tensorIds[name];
    if (typeof info === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return info.type;
  }

  _addOpsAndParams() {
    const graph = this._onnxModel.graph;
    for (let i = 0; i < graph.node.length; ++i) {
      const node = graph.node[i];
      console.log(`opType: ${node.opType}`);
      let opCode;
      let inputs = [];
      let outputs = [];
      switch(node.opType) {
        case 'Conv': {
          // Add inputs
          console.log(`  inputs: [${node.input}]`);
          const x = node.input[0];
          const w = node.input[1];
          const b = node.input[2];
          inputs.push(this._getTensorIdByName(x));
          inputs.push(this._getTensorIdByName(w));
          inputs.push(this._getTensorIdByName(b));

          const attributes = node.attribute;
          const kernelShape = getObjectByName(attributes, 'kernel_shape');
          if (!kernelShape || kernelShape.ints.length !== 2)
            throw new Error('Invalid kernelShape');
          const kernelHeight = kernelShape.ints[0];
          const kernelWidth = kernelShape.ints[1];

          const pads = getObjectByName(attributes, 'pads');
          if (!pads || pads.ints.length !== 4)
            throw new Error('Invalid pads');
          console.log(`  pads: [${pads.ints}]`);
          const paddingHeightBegin = pads.ints[0];
          const paddingWidthBegin = pads.ints[1];
          const paddingHeightEnd = pads.ints[2];
          const paddingWidthEnd = pads.ints[3]
          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));

          const strides = getObjectByName(attributes, 'strides');
          if (!strides || strides.ints.length !== 2)
            throw new Error('Invalid strides');
          console.log(`  strides: [${strides.ints}]`);
          const strideY = strides.ints[0];
          const strideX = strides.ints[1];
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));

          const nextNode = graph.node[i+1];
          let output = node.output[0];
          if (nextNode.opType === 'Relu' && node.output[0] === nextNode.input[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            console.log(`  fuse relu: ${nextNode.output[0]} -> ${output}`);
            output = nextNode.output[0];
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const inputType = this._getTensorTypeByName(x);
          const batch = inputType.dimensions[0];
          const inputHeight = inputType.dimensions[1];
          const inputWidth = inputType.dimensions[2];
          const outputHeight = Math.floor((inputHeight - kernelHeight + paddingHeightBegin + paddingHeightEnd)/strideY + 1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const filterType = this._getTensorTypeByName(w);
          const filterNum = filterType.dimensions[0];
          const outputDims = [batch, outputHeight, outputWidth, filterNum];
          let operandType = {type: this._nn.TENSOR_FLOAT32, dimensions: outputDims};
          let outputId = this._operandIndex++;
          this._tensorIds[output] = {id: outputId, type: operandType};
          this._model.addOperand(operandType);
          outputs.push(outputId);
          console.log(`  output ${output}: [${outputDims}]`);

          opCode = this._nn.CONV_2D;
        } break;
        case 'Relu': {
          throw new Error('Relu should be fused into Conv');
        } break;
        case 'MaxPool': {
          console.log(`  inputs: [${node.input}]`);
          const x = node.input[0];
          inputs.push(this._getTensorIdByName(x));

          const attributes = node.attribute;
          const pads = getObjectByName(attributes, 'pads');
          if (!pads || pads.ints.length !== 4)
            throw new Error('Invalid pads');
          console.log(`  pads: [${pads.ints}]`);
          const paddingHeightBegin = pads.ints[0];
          const paddingWidthBegin = pads.ints[1];
          const paddingHeightEnd = pads.ints[2];
          const paddingWidthEnd = pads.ints[3]
          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));

          const strides = getObjectByName(attributes, 'strides');
          if (!strides || strides.ints.length !== 2)
            throw new Error('Invalid strides');
          console.log(`  strides: [${strides.ints}]`);
          const strideY = strides.ints[0];
          const strideX = strides.ints[1];
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));

          const kernelShape = getObjectByName(attributes, 'kernel_shape');
          if (!kernelShape || kernelShape.ints.length !== 2)
            throw new Error('Invalid kernelShape');
          const kernelHeight = kernelShape.ints[0];
          const kernelWidth = kernelShape.ints[1];
          inputs.push(this._addScalarInt32(kernelWidth));
          inputs.push(this._addScalarInt32(kernelHeight));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          // Add outputs
          const output = node.output[0];
          const inputType = this._getTensorTypeByName(x);
          const batch = inputType.dimensions[0];
          const inputHeight = inputType.dimensions[1];
          const inputWidth = inputType.dimensions[2];
          const inputChannels = inputType.dimensions[3];
          const outputHeight = Math.floor((inputHeight - kernelHeight + paddingHeightBegin + paddingHeightEnd)/strideY + 1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const outputDims = [batch, outputHeight, outputWidth, inputChannels];
          let operandType = {type: this._nn.TENSOR_FLOAT32, dimensions: outputDims};
          let outputId = this._operandIndex++;
          this._tensorIds[output] = {id: outputId, type: operandType};
          this._model.addOperand(operandType);
          outputs.push(outputId);
          console.log(`  output ${output}: [${outputDims}]`);

          opCode = this._nn.MAX_POOL_2D;
        } break;
        case 'Concat': {
          console.log(`  inputs: [${node.input}]`);
          for (let i = 0; i < node.input.length; ++i) {
            inputs.push(this._getTensorIdByName(node.input[i]));
          }
          const attributes = node.attribute;
          const axis = getObjectByName(attributes, 'axis');
          if (axis && axis.i !== 1)
            throw new Error('Invalid axis ${axis.i}');
          console.log(`  axis: ${axis.i}]`);
          // C axis is 3 in NHWC layout
          const concatAxis = 3;
          inputs.push(this._addScalarInt32(concatAxis));

          // Add output
          const output = node.output[0];
          const input0Type = this._getTensorTypeByName(node.input[0]);
          let outputDims = Array.from(input0Type.dimensions);
          for (let i = 1; i < node.input.length; ++i) {
            const inputType = this._getTensorTypeByName(node.input[i]);
            outputDims[concatAxis] += inputType.dimensions[concatAxis];
          }
          let operandType = {type: this._nn.TENSOR_FLOAT32, dimensions: outputDims};
          let outputId = this._operandIndex++;
          this._tensorIds[output] = {id: outputId, type: operandType};
          this._model.addOperand(operandType);
          outputs.push(outputId);
          console.log(`  output ${output}: [${outputDims}]`);

          opCode = this._nn.CONCATENATION;
        } break;
        case 'Dropout': {
          console.log(`Skip Dropout: ${node.input[0]} -> ${node.output[0]}`);
          this._tensorIds[node.output[0]] = this._tensorIds[node.input[0]];
        } break;
        case 'GlobalAveragePool': {
          console.log(`  inputs: [${node.input}]`);
          const x = node.input[0];
          inputs.push(this._getTensorIdByName(x));
          // paddings
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          // strides
          inputs.push(this._addScalarInt32(1));
          inputs.push(this._addScalarInt32(1));
          // filters
          const inputType = this._getTensorTypeByName(x);
          const batch = inputType.dimensions[0];
          const inputHeight = inputType.dimensions[1];
          const inputWidth = inputType.dimensions[2];
          const inputChannels = inputType.dimensions[3];
          const kernelHeight = inputHeight;
          const kernelWidth = inputWidth;
          inputs.push(this._addScalarInt32(inputWidth));
          inputs.push(this._addScalarInt32(inputHeight));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          // Add outputs
          const output = node.output[0];
          const outputHeight = 1;
          const outputWidth = 1;
          const outputDims = [batch, outputHeight, outputWidth, inputChannels];
          let operandType = {type: this._nn.TENSOR_FLOAT32, dimensions: outputDims};
          let outputId = this._operandIndex++;
          this._tensorIds[output] = {id: outputId, type: operandType};
          this._model.addOperand(operandType);
          outputs.push(outputId);
          console.log(`  output ${output}: [${outputDims}]`);

          opCode = this._nn.AVERAGE_POOL_2D;
        } break;
        case 'Softmax': {
          console.log(`  inputs: [${node.input}]`);
          const x = node.input[0];
          inputs.push(this._getTensorIdByName(x));
          // Set beta to 1.0
          inputs.push(this._addScalarFloat32(1.0));
          const output = node.output[0];
          outputs.push(this._getTensorIdByName(output));

          opCode = this._nn.SOFTMAX;
        } break;
        default: {
          console.warn(`    ${node.opType} is not supported.}`);
        }
      }
      if (opCode)
        this._model.addOperation(opCode, inputs, outputs);
    }
  }
}
