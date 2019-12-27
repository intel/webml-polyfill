class OpenVINOModelImporter {
  constructor(kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._rawModel = kwargs.rawModel;
    this._model = null;
    this._compilation = null;
    this._execution = null;
    this._tensorIds = [];  // map from tensor name (graphId) to nn tensor id
    this._tensorTypes = [];
    this._operations = [];
    this._operands = [];
    this._requiredOps = new Set();
    this._options = {
      softmax: kwargs.softmax,
    };
    this._operandIndex = 0;
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
    let options = {
      backend: this._backend,
      eager: eager || false,
      supportedOps: supportedOps,
    };
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

  async * layerIterator(inputTensors, layerList) {
    const graph = this._rawModel.graphs[0];
    const getLayerOutput = async (lastNodeIdx) => {
      this._tensorIds = [];
      this._tensorTypes = [];
      this._operations = [];
      this._operands = [];
      this._operandIndex = 0;
      if (this._backend !== 'WebML' && this._compilation) {
        this._compilation._preparedModel._deleteAll();
      }

      this._model = await this._nn.createModel({backend: this._backend});
      this._addTensorOperands();
      lastNodeIdx = this._addOpsAndParams(lastNodeIdx);

      const lastNode = graph.nodes[lastNodeIdx];
      const output = lastNode.outputs[0];
      const inputIds = graph.inputs.map((input) => this._getTensorId(input));
      const outputIds = [this._getTensorId(output)];
      this._model.identifyInputsAndOutputs(inputIds, outputIds);

      await this._model.finish();
      this._compilation = await this._model.createCompilation();
      this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
      await this._compilation.finish();
      this._execution = await this._compilation.createExecution();

      const outputSize = output.shape().reduce((a, b) => a * b);
      let outputTensor;
      if (this._isQuantized) {
        outputTensor = new Uint8Array(outputSize);
      } else {
        outputTensor = new Float32Array(outputSize);
      }
      await this.compute(inputTensors, [outputTensor]);
      return {
        layerId: lastNodeIdx, outputName: lastNode.name, tensor: outputTensor,
        outputIds: outputIds, inputIds: inputIds
      };
    };

    const operatorsLength = graph.nodes.length;
    if (typeof layerList === 'undefined') {
      for (let lastNode = 0; lastNode < operatorsLength;) {
        const layerOutput = await getLayerOutput(lastNode);
        yield layerOutput;
        lastNode = layerOutput.layerId + 1;
      }
    } else {
      for (let layerId of layerList) {
        if (layerId >= operatorsLength || layerId < 0) {
          throw new Error(`Illegal layer ${layerId}`);
        }
        yield await getLayerOutput(layerId);
      }
    }
  }

  _addInputsOutputs() {
    let graph = this._rawModel.graphs[0];
    let inputs = graph.inputs.map((input) => this._getTensorId(input));
    let outputs = graph.outputs.map((output) => this._getTensorId(output));
    if (outputs.length === 0) {
      outputs = [this._getTensorId(graph.nodes[graph.nodes.length-1].outputs[0])];
    }
    if (this._options.softmax &&
        graph.nodes[graph.nodes.length-1].operator !== 'SoftMax') {
      outputs = [this._tensorIds['softmax_appended']];
    }
    this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  _addOperand(type, value) {
    let index = this._operandIndex++;
    // Cache operand type. It could be modified later: Reshape
    this._tensorTypes.push(type);
    if (typeof value !== 'undefined') {
      this._setOperandValue(index, value);
    }
    return index;
  }

  _addNamedOperand(name, type, value) {
    let index = this._addOperand(type, value);
    this._tensorIds[name] = index;
    return index;
  }

  _setOperandValue(index, value) {
    // Cache operand value. It depends on operands that have not yet been added
    this._operands[index] = value;
  }

  _addOperation(opCode, inputs, outputs) {
    console.log(`  opCode: ${opCode}`);
    console.log(`  inputs: [${inputs}], outputs: [${outputs}]`);
    // Cache operaion. It depends on operands that have not yet been added
    this._operations.push([opCode, inputs, outputs]);
    this._requiredOps.add(opCode);
  }

  _addScalarInt32(value) {
    return this._addOperand({type: this._nn.INT32}, new Int32Array([value]));
  }

  _addScalarFloat32(value) {
    return this._addOperand({type: this._nn.FLOAT32}, new Float32Array([value]));
  }

  _addTensorFloat32(tensor, dims) {
    if (tensor.constructor !== Float32Array) {
      tensor = new Float32Array(tensor);
    }
    return this._addOperand({
      type: this._nn.TENSOR_FLOAT32,
      dimensions: dims
    }, tensor);
  }

  _addTensorInt32(tensor, dims) {
    if (tensor.constructor !== Int32Array) {
      tensor = new Int32Array(tensor);
    }
    return this._addOperand({
      type: this._nn.TENSOR_INT32,
      dimensions: dims
    }, tensor);
  }

  _getOperandValue(id) {
    return this._operands[id];
  }

  _getTensorId(arg) {
    const name = arg.graphId();
    if (!this._tensorIds.hasOwnProperty(name)) {
      throw new Error(`Tensor ${name} is not found`);
    }
    return this._tensorIds[name];
  }

  _getFuseCode(node) {
    switch (node.operator) {
      case 'ReLU':
        return this._nn.FUSED_RELU;
      case 'Clamp':
        const max = node.getInt('max');
        const min = node.getInt('min');
        if (max === 6 && min === 0) {
          return this._nn.FUSED_RELU6;
        } else if (max === 1 && min === -1) {
          return this._nn.FUSED_RELU1;
        } else {
          throw new Error(`Clamp with [${min}, ${max}] cannot be fused`);
        }
    }
  }

  _getTypeCode(dataType) {
    let type;
    switch (dataType) {
      case 'float32': {
        type = this._nn.TENSOR_FLOAT32;
      } break;
      case 'I32': {
        type = this._nn.TENSOR_INT32;
      } break;
      default: {
        throw new Error(`Tensor type ${dataType} is not supported.`);
      }
    }
    return type;
  }

  // add graph inputs
  _addTensorOperands() {
    const graph = this._rawModel.graphs[0];

    for (const input of graph.inputs) {
      const inputName = input.graphId();
      const inputType = {
        type: this._getTypeCode(input.dataType()), dimensions: input.shape()
      };
      this._addNamedOperand(inputName, inputType);
    }
  }

  _addOpsAndParams(lastNode) {
    const graph = this._rawModel.graphs[0];
    let i;
    if (typeof lastNode === 'undefined') {
      lastNode = graph.nodes.length - 1;
    }
    for (i = 0; i <= lastNode; ++i) {
      let node = graph.nodes[i];
      console.log(`${node.operator} (${node.name})`);
      let opCode;
      let inputs = [];
      let outputs = [];
      switch(node.operator) {
        case 'Convolution': {
          // Add inputs
          const input = node.inputs[0];
          const convFilter = node.inputs[1];
          const convBias = node.inputs[2];

          const convFilterDims = node.getKernelShape();
          const outChannels = convFilterDims[0];
          const convFilterTensor = convFilter.getInitializer(convFilterDims);
          const convBiasTensor = typeof convBias !== 'undefined' ?
              convBias.getInitializer([outChannels]):
              new Array(outChannels).fill(0);
          console.log(`  input shape: [${input.shape()}]`);
          console.log(`  kernel shape: [${convFilterDims}]`);

          const pads_begin = node.getInts('pads_begin', [0, 0]);
          const pads_end = node.getInts('pads_end', [0, 0]);
          const [paddingHeightBegin, paddingWidthBegin] = pads_begin;
          const [paddingHeightEnd, paddingWidthEnd] = pads_end;
          console.log(`  pads begin: [${pads_begin}]`);
          console.log(`  pads end: [${pads_end}]`);

          const strides = node.getInts('strides', [1, 1]);
          const [strideY, strideX] = strides;
          console.log(`  strides: [${strides}]`);

          // reshape kernel for depthwise conv
          const inDims = node.inputs[0].shape();
          const inChannels = inDims[inDims.length-1];
          const groups = node.getInt('group', 1);
          let isDepthWiseConv = false;
          if (groups > 1) {
            if (groups !== inChannels) {
              throw new Error('Group convolution is not supported.');
            } else {
              isDepthWiseConv = true;
              console.log(`  groups: ${groups} (depthwise convolution)`);
              const nhwcData = convFilterTensor;
              const chwnData = new Float32Array(nhwcData.length);
              const N = convFilterDims[0];
              const H = convFilterDims[1];
              const W = convFilterDims[2];
              // NHWC -> CHWN where C === 1
              for (let n = 0; n < N; ++n) {
                for (let h = 0; h < H; ++h) {
                  for (let w = 0; w < W; ++w) {
                    chwnData[h*W*N + w*N + n] = nhwcData[n*H*W + h*W + w];
                  }
                }
              }
              convFilterTensor.set(chwnData);
              convFilterDims[0] = 1;
              convFilterDims[3] = groups;
            }
          }

          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(convFilterTensor, convFilterDims));
          inputs.push(this._addTensorFloat32(convBiasTensor, [outChannels]));
          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));
          if (isDepthWiseConv) {
            inputs.push(this._addScalarInt32(1)); // depth multiplier
          }

          let output = node.outputs[0];
          let nextNode = graph.nodes[i+1];
          if (nextNode && ['Clamp', 'ReLU'].includes(nextNode.operator) &&
              node.outputs[0].graphId() === nextNode.inputs[0].graphId()) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._getFuseCode(nextNode)));
            i++;
            console.log(`  fuse relu: output of ${nextNode.name}->${node.name}`);
            output = nextNode.outputs[0];
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
        } break;
        case 'Eltwise': {
          // Add inputs
          const in1 = node.inputs[0];
          const in2 = node.inputs[1];
          inputs.push(this._getTensorId(in1));
          inputs.push(this._getTensorId(in2));
          console.log(`  inputs shape: ` +
              `[${node.inputs.map((input) => input.shape()).join('], [')}]`);

          const operation = node.getString('operation');
          console.log(`  operation: ${operation}`);
          switch (operation) {
            case 'sum':
              opCode = this._nn.ADD;
              break;
            case 'mul':
              opCode = this._nn.MUL;
              break;
            default:
              throw new Error(`Operation ${operation} is not supported`);
          }

          let output = node.outputs[0];
          const nextNode = graph.nodes[i+1];
          if (nextNode && ['Clamp', 'ReLU'].includes(nextNode.operator) &&
              node.outputs[0].graphId() === nextNode.inputs[0].graphId()) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._getFuseCode(nextNode)));
            i++;
            console.log(`  fuse relu: output of ${nextNode.name}->${node.name}`);
            output = nextNode.outputs[0];
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);
        } break;
        case 'FullyConnected': {
          // Add inputs
          const input = node.inputs[0];
          const weights = node.inputs[1];
          const bias = node.inputs[2];

          const inDims = input.shape();
          let inSize = 0;
          if (inDims.length === 4) inSize = inDims[1] * inDims[2] * inDims[3];
          if (inDims.length === 3) inSize = inDims[1] * inDims[2];
          if (inDims.length === 2) inSize = inDims[1];
          const outputSize = node.getInt('out-size');
          const weightsDims = [outputSize, inSize];
          const weightsTensor = weights.getInitializer();
          const biasTensor = bias ? bias.getInitializer() : new Float32Array(outputSize).fill(0);
          console.log(`  input shape: [${inDims}]`);
          console.log(`  weights shape: [${weightsDims}]`);
          console.log(`  bias shape: [${outputSize}]`);

          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(weightsTensor, weightsDims));
          inputs.push(this._addTensorFloat32(biasTensor, [outputSize]));

          let output = node.outputs[0];
          let nextNode = graph.nodes[i+1];
          if (nextNode && ['Clamp', 'ReLU'].includes(nextNode.operator) &&
              node.outputs[0].graphId() === nextNode.inputs[0].graphId()) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._getFuseCode(nextNode)));
            i++;
            console.log(`  fuse relu: output of ${nextNode.name}->${node.name}`);
            output = nextNode.outputs[0];
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.FULLY_CONNECTED;
        } break;
        case 'ScaleShift': {
          // ScaleShift is split into Mul and Add
          const input = node.inputs[0];
          const inDims = input.shape();
          console.log(`  input shape: [${inDims}]`);

          const weights = node.inputs[1];
          const bias = node.inputs[2];
          const weightsTensor = weights.getInitializer();
          const biasTensor = bias.getInitializer();
          // put length into channel of NHWC
          const dims = [1, 1, 1, weightsTensor.length];

          // add intputs for Mul
          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(weightsTensor, dims));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          // create intermediate output for Mul
          const mulOutputType = {type: this._nn.TENSOR_FLOAT32, dimensions: inDims};
          const mulOutputId = this._addOperand(mulOutputType);
          outputs.push(mulOutputId);

          // push the Mul op
          this._addOperation(this._nn.MUL, inputs, outputs);

          // link Mul's output to Add's input
          inputs = [];
          inputs.push(mulOutputId);
          inputs.push(this._addTensorFloat32(biasTensor, dims));

          let output = node.outputs[0];
          let nextNode = graph.nodes[i+1];
          if (nextNode && ['Clamp', 'ReLU'].includes(nextNode.operator) &&
              node.outputs[0].graphId() === nextNode.inputs[0].graphId()) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._getFuseCode(nextNode)));
            i++;
            console.log(`  fuse relu: output of ${nextNode.name}->${node.name}`);
            output = nextNode.outputs[0];
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // use output of current node as Add's output
          outputs = [];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.ADD;
        } break;
        case 'Pooling': {
          const input = node.inputs[0];
          const inDims = input.shape();
          console.log(`  input shape: [${inDims}]`);

          const poolMethod = node.getString('pool-method');
          console.log(`  pool method: ${poolMethod}`);

          const strides = node.getInts('strides', [1, 1]);
          const [strideY, strideX] = strides;
          console.log(`  strides: [${strides}]`);

          const kernelShape = node.getInts('kernel');
          if (!kernelShape || kernelShape.length !== 2) {
            throw new Error(`Invalid kernel shape [${kernelShape}]`);
          }
          const kernelHeight = kernelShape[0];
          const kernelWidth = kernelShape[1];
          console.log(`  kernel shape: [${kernelShape}]`);

          const pads_begin = node.getInts('pads_begin', [0, 0]);
          const pads_end = node.getInts('pads_end', [0, 0]);
          let [padHeightBegin, padWidthBegin] = pads_begin;
          let [padHeightEnd, padWidthEnd] = pads_end;
          console.log(`  pads begin: [${pads_begin}]`);
          console.log(`  pads end: [${pads_end}]`);

          const roundingType = node.getString('rounding_type');
          console.log(`  rounding type: ${roundingType}`);
          // some caffe models uses ceil-mode padding, but we only support the
          // floor-mode padding. So we ajust the padding on both sides to make
          // it compatible but it's not equivalent to ceil-mode padding
          if (roundingType === 'ceil' &&
              (inDims[1]-kernelHeight+padHeightBegin+padHeightEnd)%strideY !== 0) {
            padHeightBegin += Math.floor(strideY / 2);
            padHeightEnd += Math.floor(strideY / 2);
            console.warn(`Ceil mode is not supported. Ajusted padHeight to ` +
                `[${padHeightBegin},${padHeightEnd}]`);
          }
          if (roundingType === 'ceil' &&
              (inDims[2]-kernelWidth+padWidthBegin+padWidthEnd)%strideX !== 0) {
            padWidthBegin += Math.floor(strideX / 2);
            padWidthEnd += Math.floor(strideX / 2);
            console.warn(`Ceil mode is not supported. Ajusted padWidth to ` +
                `[${padWidthBegin},${padWidthEnd}]`);
          }

          // zero values in the padding are not used if exclude-pad is "true"
          const excludePad = node.getBool('exclude-pad', true);
          console.log(`  exclude pad: ${excludePad}`);
          if (!excludePad &&
              (padHeightBegin || padHeightEnd || padWidthBegin || padWidthEnd)) {
            console.warn('Non execude-pad is not supported');
          }

          inputs.push(this._getTensorId(input));
          inputs.push(this._addScalarInt32(padWidthBegin));
          inputs.push(this._addScalarInt32(padWidthEnd));
          inputs.push(this._addScalarInt32(padHeightBegin));
          inputs.push(this._addScalarInt32(padHeightEnd));
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));
          inputs.push(this._addScalarInt32(kernelWidth));
          inputs.push(this._addScalarInt32(kernelHeight));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          // Add output
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          if (poolMethod === 'max') {
            opCode = this._nn.MAX_POOL_2D;
          } else if (poolMethod === 'avg') {
            opCode = this._nn.AVERAGE_POOL_2D;
          } else {
            throw new Error(`Invalid pooling method ${poolMethod}`);
          }
        } break;
        case 'Concat': {
          console.log(`  inputs shape: ` +
              `[${node.inputs.map((input) => input.shape()).join('], [')}]`);
          for (let i = 0; i < node.inputs.length; ++i) {
            inputs.push(this._getTensorId(node.inputs[i]));
          }

          const axis = node.getInts('axis');
          if (axis && (axis > 3 || axis < 0)) {
            throw new Error(`Invalid axis ${axis}`);
          }

          const input0Dims = node.inputs[0].shape();
          let concatAxis = axis;
          if (input0Dims.length === 4) {
            // NCHW -> NHWC
            concatAxis = {
              0: 0,
              1: 3,
              2: 1,
              3: 2,
            }[axis];
          }
          inputs.push(this._addScalarInt32(concatAxis));
          console.log(`  concatAxis: ${concatAxis}`);

          // Add output
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.CONCATENATION;
        } break;
        case 'Permute': {
          const input = node.inputs[0];
          const order = node.getInts('order');
          const inDims = input.shape();
          const inputId = this._getTensorId(input);
          const output = node.outputs[0];
          const outputName = output.graphId();
          if (order.toString() === '0,2,3,1') {
            this._tensorIds[outputName] = inputId;
            // equivalent to NCHW -> NHWC
            console.log(`  skip permuting to ${order.toString()}`);
          } else {
            if (order.length === 4) {
              console.log(`  input shape: [${inDims}]`);

              // Converte order data: NCHW -> NHWC
              let orderTmp = [];
              for (let i = 0; i < order.length; i++) {
                if (order[i] === 0) {
                  orderTmp[i] = order[i];
                } else if (order[i] === 1) {
                  orderTmp[i] = 3;
                } else if (order[i] === 2) {
                  orderTmp[i] = 1;
                } else {
                  orderTmp[i] = 2;
                }
              }

              // Converte order data format: NCHW -> NHWC
              const newOrder = [orderTmp[0], orderTmp[2], orderTmp[3], orderTmp[1]];

              inputs.push(inputId);
              inputs.push(this._addTensorInt32(newOrder, [4]));

              const outDims = output.shape();
              const outputType = {
                type: this._getTypeCode(output.dataType()), dimensions: outDims
              };
              const outputId = this._addNamedOperand(outputName, outputType);
              outputs.push(outputId);
              console.log(`  output shape: [${outDims}]`);

              this._addOperation(this._nn.TRANSPOSE, inputs, outputs);
            } else {
              throw new Error(`Permuting to ${order} is not supported`);
            }
          }
        } break;
        case 'Const': {
          // initializer is contained in the node
          const data = node.getInitializer();
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          this._addNamedOperand(output.graphId(), outputType, data);
          console.log(`  output shape: [${outDims}]`);
        } break;
        case 'Reshape': {
          const input = node.inputs[0];
          console.log(`  input shape: [${input.shape()}]`);

          const shape = node.inputs[1];
          const shapeId = this._getTensorId(shape);
          // `Reshape` requires `shape` to be integer. However, `shape` tensor
          // in the OpenVINO model is of type float. So we modify the type
          this._tensorTypes[shapeId].type = this._nn.TENSOR_INT32;
          const output = node.outputs[0];
          const newShape = new Int32Array(output.shape());
          this._setOperandValue(shapeId, newShape);

          inputs.push(this._getTensorId(input));
          inputs.push(shapeId);

          // Add outputs
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.RESHAPE;
        } break;
        case 'SoftMax': {
          const input = node.inputs[0];
          console.log(`  input shape: [${input.shape()}]`);

          inputs.push(this._getTensorId(input));
          inputs.push(this._addScalarFloat32(1.0)); // Set beta to 1.0

          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.SOFTMAX;
        } break;
        case 'PReLU': {
          // Now, we don't support 'channel_shared' parameter of OpenVINO's PReLU.
          const input = node.inputs[0];
          const inDims = input.shape();
          console.log(`  input shape: [${input.shape()}]`);

          const slope = node.inputs[1];
          const slopeTensor = slope.getInitializer();
          const slopeDims = [slopeTensor.length];
          console.log(`  slope shape: [${slopeDims}]`);

          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(slopeTensor, slopeDims));

          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.PRELU;
        } break;
        case 'Activation': {
          // Add inputs
          const in1 = node.inputs[0];
          inputs.push(this._getTensorId(in1));
          console.log(`  inputs shape: ` +
              `[${node.inputs.map((input) => input.shape()).join('], [')}]`);

          const type = node.getString('type');
          console.log(`  type: ${type}`);
          switch (type) {
            case 'sigmoid':
              opCode = this._nn.LOGISTIC;
              break;
            default:
              throw new Error(`The type ${type} of Activation is not supported`);
          }

          // Add outputs
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);
        } break;
        default: {
          throw new Error(`${node.operator} is not supported.`);
        }
      }

      // skip NOP, e.g. Permute
      if (typeof opCode === 'undefined') {
        continue;
      }

      if (i === graph.nodes.length - 1 && this._options.softmax &&
          node.operator !== 'SoftMax') {
        this._addOperation(opCode, inputs, outputs);

        // Add inputs
        inputs = [];
        inputs.push(outputs[0]);
        inputs.push(this._addScalarFloat32(1.0)); // Set beta to 1.0

        const inDims = node.outputs[0].shape();
        console.log(`SoftMax (appended automatically)`);
        console.log(`  input shape: [${inDims}]`);

        // Add outputs
        outputs = [];
        const outputType = {type: this._nn.TENSOR_FLOAT32, dimensions: inDims};
        const outputId = this._addNamedOperand('softmax_appended', outputType);
        outputs.push(outputId);

        opCode = this._nn.SOFTMAX;
      }

      this._addOperation(opCode, inputs, outputs);
    }

    // Write back all cached operands and operations
    for (const type of this._tensorTypes) {
      this._model.addOperand(type);
    }
    for (const [index, value] of Object.entries(this._operands)) {
      this._model.setOperandValue(index, value);
    }
    for (const [opCode, inputs, outputs] of this._operations) {
      this._model.addOperation(opCode, inputs, outputs);
    }
    return i - 1;
  }

  async getRequiredOps() {
    return this._requiredOps;
  }
}
