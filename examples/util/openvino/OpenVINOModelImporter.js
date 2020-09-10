class OpenVINOModelImporter {
  constructor(kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._isIE = kwargs.isIE;
    this._isDNN = kwargs.isDNN;
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
    this._inputScaleFactor = kwargs.inputScaleFactor;
    if (this._backend === 'WebML') {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL' || this._backend === 'WebGPU') {
      this._nn = nnPolyfill;
    }
    this._bEagerMode = false;
    this._supportedOps = new Set();
  }

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  async createCompiledModel() {
    let options = {
      backend: this._backend,
      eager: this._bEagerMode,
      supportedOps: this._supportedOps,
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
    const getLayerOutput = async (lastNodeIndex) => {
      this._tensorIds = [];
      this._tensorTypes = [];
      this._operations = [];
      this._operands = [];
      this._operandIndex = 0;
      if (this._backend !== 'WebML' && this._compilation) {
        this._compilation._preparedModel._deleteAll();
      }

      this._model = await this._nn.createModel({ backend: this._backend });
      this._addTensorOperands();
      lastNodeIndex = this._addOpsAndParams(lastNodeIndex);
      const lastNode = graph.nodes[lastNodeIndex];
      let output;
      if (lastNode.type === 'Result') {
        output = lastNode.inputs[0];
      } else {
        output = lastNode.outputs[0];
      }
      const inputIds = graph.inputs.map((input) => this._getTensorId(input));
      let outputIds = [this._getTensorId(output)];
      if (lastNodeIndex === graph.nodes.length - 2 && this._options.softmax &&
        lastNode.operator !== 'SoftMax') {
        outputsIds = [this._tensorIds['softmax_appended']];
      }
      this._model.identifyInputsAndOutputs(inputIds, outputIds);

      await this._model.finish();
      this._compilation = await this._model.createCompilation();
      this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
      await this._compilation.finish();
      this._execution = await this._compilation.createExecution();

      const outputSize = output.shape().reduce((a, b) => a * b);
      let outputTensor;
      const typedArray = this._isQuantized || false ? (this._isDNNL || this._isIE || false ? Float32Array : Uint8Array) : Float32Array;
      outputTensor = new typedArray(outputSize);
      await this.compute(inputTensors, [outputTensor]);
      return {
        layerId: lastNodeIndex, outputName: lastNode.name, tensor: outputTensor,
        outputIds: outputIds, inputIds: inputIds
      };
    };

    const operatorsLength = graph.nodes.length;
    if (typeof layerList === 'undefined') {
      for (let lastNodeIndex = 0; lastNodeIndex < operatorsLength;) {
        const layerOutput = await getLayerOutput(lastNodeIndex);
        yield layerOutput;
        lastNodeIndex = layerOutput.layerId + 1;
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
    // Set Result node's input as the model's output.
    let outputs = [this._getTensorId(graph.nodes[graph.nodes.length - 1].inputs[0])];
    if (this._options.softmax &&
      graph.nodes[graph.nodes.length - 2].operator !== 'SoftMax') {
      outputs = [this._tensorIds['softmax_appended']];
    }
    this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  _addOperand(type, value) {
    let index = this._operandIndex++;
    // Cache operand type. It could be modified later: Reshape
    if (type.type == this._nn.TENSOR_QUANT8_ASYMM) {
      type.scale = 1;
      type.zeroPoint = 0;
    }
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
    return this._addOperand({ type: this._nn.INT32 }, new Int32Array([value]));
  }

  _addScalarFloat32(value) {
    return this._addOperand({ type: this._nn.FLOAT32 }, new Float32Array([value]));
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
  // Get operand index by its name
  _getTensorId(arg) {
    const name = arg.graphId();
    console.log(`  input tensor: ${name}`);
    if (!this._tensorIds.hasOwnProperty(name)) {
      throw new Error(`Tensor ${name} is not found`);
    }
    return this._tensorIds[name];
  }

  _getFuseCode(node) {
    switch (node.type) {
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
      case 'int64':
      case 'I32': {
        type = this._nn.TENSOR_INT32;
      } break;
      case 'uint8': {
        type = this._nn.TENSOR_QUANT8_ASYMM;
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
      console.log(`${inputName}`);
      const scale = this._inputScaleFactor == undefined ? 1.0 : this._inputScaleFactor;
      const inputType = {
        type: this._getTypeCode(input.dataType()), dimensions: input.shape(), scale
      };
      this._addNamedOperand(inputName, inputType);
    }
  }


  _addOpsAndParams(lastNodeIndex) {
    const graph = this._rawModel.graphs[0];
    let i;
    if (typeof lastNodeIndex === 'undefined') {
      lastNodeIndex = graph.nodes.length - 1;
    }
    console.log(`lastNodeIndex:${lastNodeIndex}`);
    for (i = 0; i <= lastNodeIndex; ++i) {
      let node = graph.nodes[i];
      console.log(`${node.type} (${node.name})`);
      let opCode;
      let inputs = [];
      let outputs = [];
      switch (node.type) {
        case 'Convolution':
        case 'GroupConvolution': {
          // Add inputs
          const input = node.inputs[0];
          inputs.push(this._getTensorId(input));
          const inDims = input.shape();
          console.log(`  input shape: [${inDims}]`);

          let output = node.outputs[0];
          const outDims = output.shape();
          const outChannels = outDims[outDims.length - 1];

          const isDepthWiseConv = (node.type === 'GroupConvolution');

          // Add weights
          const convFilter = node.inputs[1];
          let convFilterDims = convFilter.shape();
          const convFilterName = convFilter.graphId();
          console.log(`  kernel shape: [${convFilterDims}]`);
          if (this._isQuantized && this._isIE) {
            if (!this._tensorIds.hasOwnProperty(convFilterName)) {
              const convFilterType = {
                type: this._getTypeCode(convFilter.dataType()), dimensions: convFilterDims
              };
              const convFilterTensor = convFilter.getInitializer(convFilterDims);
              inputs.push(this._addOperand(convFilterType, convFilterTensor));
            } else {
              inputs.push(this._getTensorId(convFilter));
            }
          } else {
            let convFilterTensor;
            if (isDepthWiseConv) {
              // Reshape GOIHW kernel for depthwise conv
              const kernelH = convFilterDims[3];
              const kernelW = convFilterDims[4];
              const inChannels = inDims[inDims.length - 1];
              const groups = inChannels;
              convFilterDims = [outChannels, kernelH, kernelW, inChannels / groups];
              convFilterTensor = convFilter.getInitializer(convFilterDims);
              const nhwcData = convFilterTensor;
              const chwnData = new Float32Array(nhwcData.length);
              const N = convFilterDims[0];
              const H = convFilterDims[1];
              const W = convFilterDims[2];
              // NHWC -> CHWN where C === 1
              for (let n = 0; n < N; ++n) {
                for (let h = 0; h < H; ++h) {
                  for (let w = 0; w < W; ++w) {
                    chwnData[h * W * N + w * N + n] = nhwcData[n * H * W + h * W + w];
                  }
                }
              }
              convFilterTensor.set(chwnData);
              convFilterDims[0] = 1;
              convFilterDims[3] = groups;
              console.log(`  reshaped kernel shape: [${convFilterDims}]`);
            } else {
              convFilterTensor = convFilter.getInitializer(convFilterDims);
            }

            const convFilterType = {
              type: this._getTypeCode(convFilter.dataType()), dimensions: convFilterDims
            };
            inputs.push(this._addOperand(convFilterType, convFilterTensor));
          }

          // Add bias
          let addNode;
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addTensorFloat32(new Array(outChannels).fill(0), [outChannels]));
            console.log(`  add bias filled with 0`);
          } else {
            addNode = graph.nodes[i + 1];
            if (addNode && addNode.type === 'Add' &&
              node.outputs[0].graphId() === addNode.inputs[0].graphId()) {
              const bias = addNode.inputs[1];
              const biasDims = bias.shape();
              const biasType = {
                type: this._getTypeCode(bias.dataType()), dimensions: biasDims
              };
              const biasTensor = bias.getInitializer(biasDims);
              inputs.push(this._addOperand(biasType, biasTensor));
              i++;
              console.log(`  add bias via ${addNode.name}->${node.name}`);
              output = addNode.outputs[0];
            } else {
              inputs.push(this._addTensorFloat32(new Array(outChannels).fill(0), [outChannels]));
              console.log(`  add bias filled with 0`);
            }
          }

          // Add attributes
          const pads_begin = node.getInts('pads_begin', '0, 0');
          const pads_end = node.getInts('pads_end', '0, 0');
          const [paddingHeightBegin, paddingWidthBegin] = pads_begin;
          const [paddingHeightEnd, paddingWidthEnd] = pads_end;
          console.log(`  pads begin: [${pads_begin}]`);
          console.log(`  pads end: [${pads_end}]`);
          const dilations = node.getInts('dilations');
          let isAtrous = false;
          let strides = null;
          if (dilations[0] !== 1 && dilations[1] !== 1) {
            strides = node.getInts('dilations');
            isAtrous = true;
          } else {
            strides = node.getInts('strides');
          }
          const [strideY, strideX] = strides;
          console.log(`  strides: [${strides}]`);
          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));
          if (isDepthWiseConv) {
            inputs.push(this._addScalarInt32(1)); // depth multiplier
          }

          // Add fused code
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          } else {
            const fusedNode = graph.nodes[i + 1];
            if (fusedNode && ['Clamp', 'ReLU'].includes(fusedNode.type) &&
              addNode.outputs[0].graphId() === fusedNode.inputs[0].graphId()) {
              // Fuse relu
              inputs.push(this._addScalarInt32(this._getFuseCode(fusedNode)));
              i++;
              console.log(`  fuse relu: output of ${fusedNode.name}->${node.name}`);
              output = fusedNode.outputs[0];
            } else {
              inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
            }
          }

          // Add output
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          if (isDepthWiseConv == true) {
            opCode = isAtrous ? this._nn.ATROUS_DEPTHWISE_CONV_2D : this._nn.DEPTHWISE_CONV_2D;
          } else {
            opCode = isAtrous ? this._nn.ATROUS_CONV_2D : this._nn.CONV_2D;
          }
        } break;
        case 'Multiply': {
          // Add inputs
          for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            let name = input.graphId();
            const inputDims = input.shape();
            if (!this._tensorIds.hasOwnProperty(name)) {
              const inputType = {
                type: this._getTypeCode(input.dataType()), dimensions: inputDims
              };
              const inputTensor = input.getInitializer(inputDims);
              inputs.push(this._addOperand(inputType, inputTensor));
            } else {
              inputs.push(this._getTensorId(input));
            }
          }

          console.log(`  inputs shape: ` +
            `[${node.inputs.map((input) => input.shape()).join('], [')}]`);

          let output = node.outputs[0];
          opCode = this._nn.MUL;

          // Add fused code
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          } else {
            const fusedNode = graph.nodes[i + 1];
            if (fusedNode && ['Clamp', 'ReLU'].includes(fusedNode.type) &&
              node.outputs[0].graphId() === fusedNode.inputs[0].graphId()) {
              // Fuse relu
              inputs.push(this._addScalarInt32(this._getFuseCode(fusedNode)));
              i++;
              console.log(`  fuse relu: output of ${fusedNode.name}->${node.name}`);
              output = fusedNode.outputs[0];
            } else {
              inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
            }
          }


          // Add output
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);
        } break;
        case 'Add': {
          // Add inputs
          for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            let name = input.graphId();
            const inputDims = input.shape();
            if (!this._tensorIds.hasOwnProperty(name)) {
              const inputType = {
                type: this._getTypeCode(input.dataType()), dimensions: inputDims
              };
              const inputTensor = input.getInitializer(inputDims);
              inputs.push(this._addOperand(inputType, inputTensor));
            } else {
              inputs.push(this._getTensorId(input));
            }
          }

          console.log(`  inputs shape: ` +
            `[${node.inputs.map((input) => input.shape()).join('], [')}]`);

          let output = node.outputs[0];
          opCode = this._nn.ADD;

          // Add fused code
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          } else {
            const fusedNode = graph.nodes[i + 1];
            if (fusedNode && ['Clamp', 'ReLU'].includes(fusedNode.type) &&
              node.outputs[0].graphId() === fusedNode.inputs[0].graphId()) {
              // Fuse relu
              inputs.push(this._addScalarInt32(this._getFuseCode(fusedNode)));
              i++;
              console.log(`  fuse relu: output of ${fusedNode.name}->${node.name}`);
              output = fusedNode.outputs[0];
            } else {
              inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
            }
          }

          // Add output
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);
        } break;
        case 'MatMul': {
          // Add inputs
          const input = node.inputs[0];
          const inDims = input.shape();
          inputs.push(this._getTensorId(input));
          console.log(`  input shape: [${inDims}]`);

          // Add weights
          const weights = node.inputs[1];
          const weightsDims = weights.shape();
          const num_units = weightsDims[0];
          const weightsName = weights.graphId();
          console.log(`  weights shape: [${weightsDims}]`);

          if (!this._tensorIds.hasOwnProperty(weightsName)) {
            const weightsType = {
              type: this._getTypeCode(weights.dataType()), dimensions: weightsDims
            };
            const weightsTensor = weights.getInitializer(weightsDims);
            inputs.push(this._addOperand(weightsType, weightsTensor));
          } else {
            inputs.push(this._getTensorId(weights));
          }

          let output = node.outputs[0];

          // Add bias
          let addNode;
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addTensorFloat32(new Array(num_units).fill(0), [num_units]));
            console.log(`  add bias filled with 0`);
          } else {
            addNode = graph.nodes[i + 1];
            if (addNode && addNode.type === 'Add' &&
              node.outputs[0].graphId() === addNode.inputs[0].graphId()) {
              const bias = addNode.inputs[1];
              const biasDims = bias.shape();
              console.log(`  bias shape: [${biasDims}]`);
              const biasType = {
                type: this._getTypeCode(bias.dataType()), dimensions: [num_units]
              };
              const biasTensor = bias.getInitializer(biasDims);
              inputs.push(this._addOperand(biasType, biasTensor));
              i++;
              console.log(`  add bias via ${addNode.name}->${node.name}`);
              output = addNode.outputs[0];
            } else {
              inputs.push(this._addTensorFloat32(new Array(num_units).fill(0), [num_units]));
              console.log(`  add bias filled with 0`);
            }
          }

          // Add fused code
          if (i + 1 > lastNodeIndex) {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          } else {
            const fusedNode = graph.nodes[i + 1];
            if (fusedNode && ['Clamp', 'ReLU'].includes(fusedNode.type) &&
              addNode.outputs[0].graphId() === fusedNode.inputs[0].graphId()) {
              // Fuse relu
              inputs.push(this._addScalarInt32(this._getFuseCode(fusedNode)));
              i++;
              console.log(`  fuse relu: output of ${fusedNode.name}->${node.name}`);
              output = fusedNode.outputs[0];
            } else {
              inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
            }
          }

          // Add output
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.FULLY_CONNECTED;
        } break;
        case 'ReduceMean':
        case 'MaxPool':
        case 'AvgPool': {
          const isReduceMean = node.type === 'ReduceMean'
          const input = node.inputs[0];
          const inDims = input.shape();
          console.log(`  input shape: [${inDims}]`);

          const poolMethod = isReduceMean ? 'avgReplaced' : node.type === 'MaxPool' ? 'max' : 'avg';
          console.log(`  pool method: ${poolMethod}`);
          const strides = node.getInts('strides', '1, 1');
          const [strideY, strideX] = strides;
          console.log(`  strides: [${strides}]`);
          const kernelShape = isReduceMean ? [inDims[1], inDims[2]] : node.getInts('kernel');
          if (!kernelShape || kernelShape.length !== 2) {
            throw new Error(`Invalid kernel shape [${kernelShape}]`);
          }
          const kernelHeight = kernelShape[0];
          const kernelWidth = kernelShape[1];
          console.log(`  kernel shape: [${kernelShape}]`);

          const pads_begin = node.getInts('pads_begin', '0, 0');
          const pads_end = node.getInts('pads_end', '0, 0');
          let [padHeightBegin, padWidthBegin] = pads_begin;
          let [padHeightEnd, padWidthEnd] = pads_end;
          console.log(`  pads begin: [${pads_begin}]`);
          console.log(`  pads end: [${pads_end}]`);
          const roundingType = isReduceMean ? 'ceil' : node.getString('rounding_type');
          console.log(`  rounding type: ${roundingType}`);
          // some caffe models uses ceil-mode padding, but we only support the
          // floor-mode padding. So we ajust the padding on both sides to make
          // it compatible but it's not equivalent to ceil-mode padding
          if (roundingType === 'ceil' &&
            (inDims[1] - kernelHeight + padHeightBegin + padHeightEnd) % strideY !== 0) {
            padHeightBegin += Math.floor(strideY / 2);
            padHeightEnd += Math.floor(strideY / 2);
            console.warn(`Ceil mode is not supported. Ajusted padHeight to ` +
              `[${padHeightBegin},${padHeightEnd}]`);
          }
          if (roundingType === 'ceil' &&
            (inDims[2] - kernelWidth + padWidthBegin + padWidthEnd) % strideX !== 0) {
            padWidthBegin += Math.floor(strideX / 2);
            padWidthEnd += Math.floor(strideX / 2);
            console.warn(`Ceil mode is not supported. Ajusted padWidth to ` +
              `[${padWidthBegin},${padWidthEnd}]`);
          }

          // zero values in the padding are not used if exclude-pad is "true"
          const excludePad = isReduceMean ? false : node.getBool('exclude-pad', true);
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
          } else if (poolMethod === 'avg' || poolMethod === 'avgReplaced') {
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
        case 'Transpose': {
          const input = node.inputs[0];
          const inDims = input.shape();
          const inputId = this._getTensorId(input);
          const transposeOrder = node.inputs[1];
          const orderTensor = transposeOrder.getInitializer()
          // const order = orderTensor.filter(x => orderTensor.indexOf(x)%2 === 0); 

          let order = [];
          for (let i = 0; i < orderTensor.length; i++) {
            if (i % 2 === 0)
              order.push(orderTensor[i])
          }

          console.log(`  transpose order: [${order}]`);
          const output = node.outputs[0];
          const outputName = output.graphId();
          if (order.toString() === '0,2,3,1') {
            this._tensorIds[outputName] = inputId;
            // equivalent to NCHW -> NHWC
            console.log(`  skip transpose to ${order.toString()}`);
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
              throw new Error(`Transpose to ${order} is not supported`);
            }
          }
        } break;
        case 'Const': {
          // Initializer is contained in the node
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
          const output = node.outputs[0];
          console.log(`  input shape: [${input.shape()}]`);
          const shape = node.inputs[1];
          const outDims = output.shape();
          const shapeType = {
            type: this._getTypeCode(shape.dataType()), dimensions: [outDims.length]
          };
          let temp = new Int32Array(outDims);
          // NGCHW -> NGHWC
          if (outDims.length == 5) {
            outDims[0] = temp[0];
            outDims[1] = temp[1];
            outDims[2] = temp[3];
            outDims[3] = temp[4];
            outDims[4] = temp[2];
          }
          const newShape = new Int32Array(outDims);
          const shapeId = this._addNamedOperand(shape.graphId(), shapeType, newShape);
          // `Reshape` requires `shape` to be integer. However, `shape` tensor
          // in the OpenVINO model is of type float. So we modify the type
          this._tensorTypes[shapeId].type = this._nn.TENSOR_INT32;
          inputs.push(this._getTensorId(input));
          inputs.push(shapeId);

          // Add output
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

          // Add output
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
          console.log(`  input shape: [${input.shape()}]`);

          const slope = node.inputs[1];
          const slopeTensor = slope.getInitializer();
          const slopeDims = [slopeTensor.length];
          console.log(`  slope shape: [${slopeDims}]`);

          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(slopeTensor, slopeDims));

          // Add output
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
        case 'Interpolate': {
          // Add inputs
          const input = node.inputs[0];
          console.log(`  input shape: [${input.shape()}]`);
          inputs.push(this._getTensorId(input));

          const output = node.outputs[0];
          const outDims = output.shape();

          // Specify the width and hight of output
          inputs.push(this._addScalarInt32(outDims[2]));
          inputs.push(this._addScalarInt32(outDims[1]));

          const align_corners = node.getInt("align_corners");
          if (align_corners !== 'undefined') {
            inputs.push(this._addScalarInt32(align_corners));
          }

          // Add output
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.RESIZE_BILINEAR;
        } break;
        case 'TopK': {
          // Add inputs
          const input = node.inputs[0];
          const inputDims = input.shape();
          console.log(`  input shape: [${inputDims}]`);
          inputs.push(this._getTensorId(input));
          const axis = node.getInt("axis");
          let argMaxAxis = axis;
          if (inputDims.length === 4) {
            // NCHW -> NHWC
            argMaxAxis = {
              0: 0,
              1: 3,
              2: 1,
              3: 2,
            }[axis];
          }
          inputs.push(this._addScalarInt32(argMaxAxis));

          // Add output
          // Outputs[1] is the index output of TopK
          const output = node.outputs[1];
          const outDims = output.shape();
          // The result has the same shape as input with the dimension along axis removed.
          outDims.splice(argMaxAxis, 1);
          const outputType = {
            type: this._getTypeCode('I32'), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.ARGMAX;
        } break;
        case 'Sigmoid': {
          opCode = this._nn.LOGISTIC;
          // Add inputs
          const input = node.inputs[0];
          inputs.push(this._getTensorId(input));

          // Add output
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);

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

          // Add output
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);
        } break;
        case 'FakeQuantize': {
          const input = node.inputs[0];
          const inputLow = node.inputs[4];
          const inputHigh = node.inputs[3];
          const outputLow = node.inputs[2];
          const outputHigh = node.inputs[1];

          let inputLowDims = inputLow.shape();
          let inputHighDims = inputHigh.shape();
          let outputLowDims = outputLow.shape();
          let outputHighDims = outputHigh.shape();

          const output = node.outputs[0];
          const outDims = output.shape();

          const inputLowTensor = inputLow.getInitializer(inputLowDims);
          const inputHighTensor = inputHigh.getInitializer(inputHighDims);
          const outputLowTensor = outputLow.getInitializer(outputLowDims);
          const outputHighTensor = outputHigh.getInitializer(outputHighDims);

          console.log(`  input shape: [${input.shape()}]`);
          inputs.push(this._getTensorId(input));
          inputs.push(this._addTensorFloat32(inputLowTensor, inputLowDims.length == 0 ? [1] : inputLowDims));
          inputs.push(this._addTensorFloat32(inputHighTensor, inputHighDims.length == 0 ? [1] : inputHighDims));
          inputs.push(this._addTensorFloat32(outputLowTensor, outputLowDims.length == 0 ? [1] : outputLowDims));
          inputs.push(this._addTensorFloat32(outputHighTensor, outputHighDims.length == 0 ? [1] : outputHighDims));

          const levels = node.getInts('levels');
          inputs.push(this._addScalarInt32(levels));

          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.FAKE_QUANTIZE;
        } break;
        case 'Convert': {
          const input = node.inputs[0];
          console.log(`  input shape: [${input.shape()}]`);

          inputs.push(this._getTensorId(input));

          // Add output
          const output = node.outputs[0];
          const outDims = output.shape();
          const outputType = {
            type: this._getTypeCode(output.dataType()), dimensions: outDims
          };
          console.log(`  outputType: ${outputType}`);
          const outputId = this._addNamedOperand(output.graphId(), outputType);
          outputs.push(outputId);
          console.log(`  output shape: [${outDims}]`);

          opCode = this._nn.CONVERT;
        } break;
        case 'Squeeze': {
          // Skip this unsupported Squeeze op
          const input = node.inputs[0];
          const inputId = this._getTensorId(input);
          const output = node.outputs[0];
          const outputName = output.graphId();
          this._tensorIds[outputName] = inputId;
          console.log(`  Skip Squeeze op`);
        } break;
        case 'Result': {
          // Result node has one input, and no output.
          const input = node.inputs[0];
          console.log(`  input shape: [${input.shape()}]`);
          inputs.push(this._getTensorId(input))
        } break;
        default: {
          throw new Error(`${node.type} is not supported.`);
        }
      }

      // skip NOP, e.g. Permute
      if (typeof opCode === 'undefined') {
        continue;
      }

      if (i === graph.nodes.length - 2 && this._options.softmax &&
        node.operator !== 'SoftMax') {
        // Add the node[i] followed by appended SoftMax
        this._addOperation(opCode, inputs, outputs);

        // Add inputs
        inputs = [];
        inputs.push(outputs[0]);
        inputs.push(this._addScalarFloat32(1.0)); // Set beta to 1.0

        const inDims = node.outputs[0].shape();
        console.log(`SoftMax (appended automatically)`);
        console.log(`  input shape: [${inDims}]`);

        // Add output
        outputs = [];
        const outputType = { type: this._nn.TENSOR_FLOAT32, dimensions: inDims };
        const outputId = this._addNamedOperand('softmax_appended', outputType);
        outputs.push(outputId);

        opCode = this._nn.SOFTMAX;
      }
      // Add the appended SoftMax if defined or add node[i] directly
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

  getRequiredOps() {
    return this._requiredOps;
  }
}
