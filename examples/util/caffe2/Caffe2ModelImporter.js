class Caffe2ModelImporter {
  constructor (kwargs) {
    this._isQuantized = kwargs.isQuantized;
    this._rawModel = kwargs.rawModel;
    this._inputSize = kwargs.inputSize;
    this._isDNNL = kwargs.isDNNL;
    this._model = null;
    this._compilation = null;
    this._execution = null;
    this._tensorIds = [];         //{name: ID}
    this._tensorTypes = [];       //{ID: type}
    this._operations = [];        //{[opCode, inputs, outputs]}
    this._operands = [];          //{ID: value}
    this._quantParams = [];       //{ID: type}
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
    } else if (this._backend === 'WASM' || this._backend === 'WebGL') {
      this._nn = nnPolyfill;
    }
    this._bEagerMode = false;
    this._supportedOps = new Set();
  }

  setEagerMode (flag) {
    this._bEagerMode = flag;
  };

  setSupportedOps (ops) {
    this._supportedOps = ops;
  };

  async createCompiledModel () {
    let options = {
      backend: this._backend,
      eager: this._bEagerMode,
      supportedOps: this._supportedOps,
    };
    this._model = await this._nn.createModel(options);
    this._setInputTensor();
    this._addOperandsAndArgs();
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

  async compute (inputTensors, outputTensors) {
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

  _setInputTensor () {
    let inputName = this._rawModel[0].input[0].name;
    let inputDims = this._inputSize.length == 3 ?
    [1, this._inputSize[0], this._inputSize[1], this._inputSize[2]] : this._inputSize;
    let inputType;
    if (this._isQuantized) {
      inputType = {
        type: null,
        dimensions: inputDims,
        scale: this._rawModel[0].arg["X_scale"]["value"],
        zeroPoint: this._isDNNL ? 0 : this._rawModel[0].arg["X_zero_point"]["value"]
      };
      inputType.type = inputType.zeroPoint == 0 ? this._nn.TENSOR_QUANT8_ASYMM_SIGNED : this._nn.TENSOR_QUANT8_ASYMM;
    } else {
      inputType = {
        type: this._nn.TENSOR_FLOAT32,
        dimensions: inputDims
      };
    }
    this._addTensor(inputName, inputType);
  }

  _addTensor (name, type, value) {
    let index = this._addOperand(type, value);
    this._tensorIds[name] = index;
    return index;
  }

  _addTensorInt32 (dime, value) {
    return this._addOperand({type: this._nn.TENSOR_INT32, dimensions: dime}, new Int32Array(value));
  }

  _addArgInt32 (value) {
    return this._addOperand({type: this._nn.INT32}, new Int32Array(value));
  }

  _addArgFloat32 (value) {
    return this._addOperand({type: this._nn.FLOAT32}, new Float32Array(value));
  }

  _addOperand (type, value) {
    let index = this._operandIndex++;
    // Cache operand type
    this._tensorTypes.push(type);
    if (typeof value !== 'undefined') {
      this._setOperandValue(index, value);
    }
    return index;
  }

  _addOperation(opCode, inputs, outputs) {
    console.log(`  opCode: ${opCode}`);
    console.log(`  inputs: [${inputs}], outputs: [${outputs}]`);
    // Cache operaion. It depends on operands that have not yet been added
    this._operations.push([opCode, inputs, outputs]);
    this._requiredOps.add(opCode);
  }

  _addInputsOutputs() {
    let inputTensor = this._rawModel[0].input[0];
    let inputName = this._getAttributeName(inputTensor);
    let outputTensor = this._rawModel[this._rawModel.length - 1].output[0];
    let outputName = this._getAttributeName(outputTensor);
    let inputs = [this._getTensorIdByName(inputName)];
    let outputs = [this._getTensorIdByName(outputName)];
    this._model.identifyInputsAndOutputs(inputs, outputs);
  }

  _setOperandValue (index, value) {
    // Cache operand value
    this._operands[index] = value;
  }

  _getAttributeName (tensor) {
    return tensor["name"];
  }

  _getAttributeValue (tensor, keyword) {
    if (typeof tensor[keyword] == "undefined") {
      return tensor[keyword + "s"]["value"];
    } else {
      return tensor[keyword]["value"];
    }
  }

  _getAttributeType (tensor, keyword) {
    if (typeof tensor[keyword] == "undefined") {
      return tensor[keyword + "s"]["type"];
    } else {
      return tensor[keyword]["type"];
    }
  }

  _getTensorIdByName (name) {
    let index = this._tensorIds[name];
    if (typeof index === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return index;
  }

  _getTensorTypeByName (name) {
    let index = this._tensorIds[name];
    if (typeof index === 'undefined')
      throw new Error(`Tensor ${name} is not found`);
    return this._tensorTypes[index];
  }

  _getFuseCode(max, min) {
    if (max == 6 && min == 0) {
      return this._nn.FUSED_RELU6;
    } else if (max == 1 && min == -1) {
      return this._nn.FUSED_RELU1;
    } else {
      return this._nn.FUSED_NONE;
    }
  }

  // Add operands
  _addOperandsAndArgs() {
    for (let nodeIdx = 0; nodeIdx < this._rawModel.length; nodeIdx++) {
      let node = this._rawModel[nodeIdx];
      console.log(`layer${nodeIdx}: ${node.operator} (${node.name})`);

      let opCode;
      let inputs = [];
      let outputs = [];
      switch(node.operator) {
        case "Conv":
        case "Int8Conv":
        case "Int8ConvRelu": {
          // Add inputs
          let inputTensor = node.input[0];
          let filterTensor = node.input[1];
          let biasTensor = node.input[2];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;
          let inputPoint = 0;
          let inputScales = 1;
          if (this._isQuantized) {
            inputPoint = inputType.zeroPoint;
            inputScales = inputType.scale;
          }
          console.log(`  input shape: [${inputDime}]`);

          // Filter
          let filterName = this._getAttributeName(filterTensor);
          let filterDims = this._getAttributeValue(filterTensor, "shape");
          let filterValue = this._getAttributeValue(filterTensor, "values");
          let filterDataType = this._getAttributeType(filterTensor, "values");
          let filterPoint = 0;
          let filterScales = 1;
          if (this._isQuantized) {
            filterPoint = this._getAttributeValue(filterTensor, "Y_zero_point");
            filterScales = this._getAttributeValue(filterTensor, "Y_scales");
          }
          let filterTypeCode = inputTypeCode;
          let isPerChannel = false;
          if (this._isQuantized && filterScales.length > 1) {
            filterTypeCode = this._nn.TENSOR_QUANT8_SYMM_PER_CHANNEL;
            isPerChannel = true;
          }

          // Bias
          let biasName = this._getAttributeName(biasTensor);
          let biasDims = this._getAttributeValue(biasTensor, "shape");
          let biasValue = this._getAttributeValue(biasTensor, "values");
          let biasDataType = this._getAttributeType(biasTensor, "values");
          let biasPoint = 0;
          let biasScales = 1;
          let biasTypeCode = this._nn.TENSOR_FLOAT32;
          if (this._isQuantized) {
            biasTypeCode = this._nn.TENSOR_INT32;
            biasPoint = this._getAttributeValue(biasTensor, "Y_zero_point");
            biasScales = this._getAttributeValue(biasTensor, "Y_scales");
          }
          let biasType = {
            type: biasTypeCode,
            dimensions: biasDims
          };
          if (this._isQuantized && !isPerChannel) {
            biasType = {
              type: biasTypeCode,
              dimensions: biasDims,
              scale: biasScales,
              zeroPoint: biasPoint
            };
          }
          console.log(`  bias shape: [${biasDims}]`);

          // Kernel
          let kernels = [];
          let kernelsTmp = this._getAttributeValue(args, "kernel");
          if (typeof kernelsTmp.length == "undefined") {
            kernels = [kernelsTmp, kernelsTmp];
          } else {
            kernels = kernelsTmp;
          }

          // Dilations
          // newKernel = dilations * (kernel - 1) + 1
          let dilations = [1, 1];
          if (args.hasOwnProperty("dilations")) {
            dilations = this._getAttributeValue(args, "dilations");
          }
          kernels = [dilations[0] * (kernels[0] - 1) + 1, dilations[1] * (kernels[1] - 1) + 1];

          if (!kernels || kernels.length !== 2)
            throw new Error("Invalid kernels");
          let kernelHeight = kernels[0];
          let kernelWidth = kernels[1];
          console.log(`  kernels: [${kernels}]`);

          // Pad
          let pads = [];
          let padsTmp = this._getAttributeValue(args, "pad");
          if (typeof padsTmp.length == "undefined") {
            pads = [padsTmp, padsTmp, padsTmp, padsTmp];
          } else {
            pads = padsTmp;
          }

          if (pads.length !== 4)
            throw new Error("Invalid pads");
          let paddingTop = pads[0];
          let paddingLeft = pads[1];
          let paddingBottom = pads[2];
          let paddingRight = pads[3];
          console.log(`  pads: [${pads}]`);

          // Stride
          let strides = [];
          let stridesTmp = this._getAttributeValue(args, "stride");
          if (typeof stridesTmp.length == "undefined") {
            strides = [stridesTmp, stridesTmp];
          } else {
            strides = stridesTmp;
          }

          if (!strides || strides.length !== 2)
            throw new Error("Invalid strides");
          let strideHeight = strides[0];
          let strideWidth = strides[1];
          console.log(`  strides: [${strides}]`);

          // Fusion type
          // 0: FUSION_UNKNOWN
          // 1: FUSION_CONV_RELU
          // 2: FUSION_CONV_BRELU
          // 3: FUSION_CONV_SUM
          // 4: FUSION_CONV_SUM_RELU
          // 5: FUSION_MAX = FUSION_CONV_SUM_RELU + 1
          let fusion = 0;
          if (args.hasOwnProperty("fusion_type")) {
            fusion = this._getAttributeValue(args, "fusion_type");
          }

          // Fuse Relu
          let nextNode = this._rawModel[nodeIdx + 1];
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let nextInputTensor = nextNode.input[0];
          let nextInputName = this._getAttributeName(nextInputTensor);
          let nextOutputTensor = nextNode.output[0];
          let nextOutputName = this._getAttributeName(nextOutputTensor);
          let fuseCode = 0;
          if (this._isDNNL) {
            // Just support 'FUSION_CONV_BRELU' for now
            if (fusion == 2) {
              let boundMax = 0;
              let boundMin = 0;
              if (args.hasOwnProperty("bound_max") && args.hasOwnProperty("bound_min")) {
                boundMax = this._getAttributeValue(args, "bound_max");
                boundMin = this._getAttributeValue(args, "bound_min");
                console.log(`  bound: [${boundMax}, ${boundMin}]`);
              }
              fuseCode = this._getFuseCode(boundMax, boundMin);
            }
          } else {
            if (nextNode && nextNode.operator === "Relu" && outputName === nextInputName) {
              // Fuse relu
              fuseCode = this._nn.FUSED_RELU;
              nodeIdx++;
              outputName = nextOutputName;
            } else {
              fuseCode = this._nn.FUSED_NONE;
            }
          }
          console.log(`  fuseCode: ${fuseCode}`);

          // Group
          let isDepthWiseConv = false;
          let group = 0;
          let inputChannel = inputDime[inputDime.length - 1];
          if (args.hasOwnProperty("group")) {
            group = this._getAttributeValue(args, "group");
          }

          if (group > 1) {
            if (group !== inputChannel) {
              throw new Error("Group convolution is not supported.");
            } else {
              isDepthWiseConv = true;
              console.log(`  group: ${group} (depthwise convolution)`);
              let nhwcData = filterValue;
              let chwnData = new Int8Array(nhwcData.length);
              let N = filterDims[0];
              let H = filterDims[1];
              let W = filterDims[2];
              // NHWC -> CHWN where C === 1
              for (let n = 0; n < N; ++n) {
                for (let h = 0; h < H; ++h) {
                  for (let w = 0; w < W; ++w) {
                    chwnData[h*W*N + w*N + n] = nhwcData[n*H*W + h*W + w];
                  }
                }
              }
              filterValue = chwnData;
              filterDims[0] = 1;
              filterDims[3] = group;
            }
          }
          console.log(`  filter shape: [${filterDims}]`);

          let filterType = {
            type: filterTypeCode,
            dimensions: filterDims
          };
          if (this._isQuantized && !isPerChannel) {
            filterType = {
              type: filterTypeCode,
              dimensions: filterDims,
              scale: filterScales,
              zeroPoint: filterPoint
            };
          }

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addTensor(filterName, filterType, filterValue));
          let filterID = this._getTensorIdByName(filterName);
          let channelDim = 0;
          if (isPerChannel) {
            if (isDepthWiseConv) {
              channelDim = 3;
            }
            this._quantParams[filterID] = {
              channelDim: channelDim,
              scales: Float32Array.from(filterScales)
            };
          }
          inputs.push(this._addTensor(biasName, biasType, biasValue));
          inputs.push(this._addArgInt32([paddingLeft]));
          inputs.push(this._addArgInt32([paddingRight]));
          inputs.push(this._addArgInt32([paddingTop]));
          inputs.push(this._addArgInt32([paddingBottom]));
          inputs.push(this._addArgInt32([strideWidth]));
          inputs.push(this._addArgInt32([strideHeight]));
          if (isDepthWiseConv) {
            inputs.push(this._addArgInt32([1]));
          }
          inputs.push(this._addArgInt32([fuseCode]));

          // Add outputs
          let outputTypeCode = inputTypeCode;
          let outputDims = [
            inputDime[0],
            Math.floor((inputDime[1] - kernelHeight + paddingTop +paddingBottom) / strideHeight + 1),
            Math.floor((inputDime[2] - kernelWidth + paddingRight + paddingLeft) / strideWidth + 1),
            biasDims[0]
          ];
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims
          };
          if (this._isQuantized) {
            let outputScales = 1;
            if (args.hasOwnProperty("Y_scale")) {
              outputScales = this._getAttributeValue(args, "Y_scale");
            }
            let outputPoint = 0;
            if (args.hasOwnProperty("Y_zero_point")) {
              outputPoint = this._getAttributeValue(args, "Y_zero_point");
            }

            if (this._isDNNL) {
              if (outputScales == 1) {
                outputTypeCode = this._nn.TENSOR_FLOAT32;
              } else if (outputPoint == 0) {
                outputTypeCode = this._nn.TENSOR_QUANT8_ASYMM;
              } else if (outputPoint == 128) {
                outputTypeCode = this._nn.TENSOR_QUANT8_ASYMM_SIGNED;
              }
            }

            outputType = {
              type: outputTypeCode,
              dimensions: outputDims,
              scale: outputScales,
              zeroPoint: outputPoint
            };
          }

          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          opCode = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
        } break;
        case "MaxPool":
        case "AveragePool":
        case "Int8AveragePool": {
          // Add inputs
          let inputTensor = node.input[0];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;
          let inputPoint = 0;
          let inputScales = 1;
          if (this._isQuantized) {
            inputPoint = inputType.zeroPoint;
            inputScales = inputType.scale;
          }
          console.log(`  input shape: [${inputDime}]`);

          // Pad
          let pads = [0, 0, 0, 0];
          if (args.hasOwnProperty("pad") || args.hasOwnProperty("pads")) {
            let padsTmp = this._getAttributeValue(args, "pad");
            if (typeof padsTmp.length == "undefined") {
              pads = [padsTmp, padsTmp, padsTmp, padsTmp];
            } else {
              pads = padsTmp;
            }
          }

          if (pads.length !== 4)
            throw new Error("Invalid pads");
          let paddingTop = pads[0];
          let paddingLeft = pads[1];
          let paddingBottom = pads[2];
          let paddingRight = pads[3];
          console.log(`  pads: [${pads}]`);

          // Stride
          let strides = [];
          let stridesTmp = this._getAttributeValue(args, "stride");
          if (typeof stridesTmp.length == "undefined") {
            strides = [stridesTmp, stridesTmp];
          } else {
            strides = stridesTmp;
          }

          if (!strides || strides.length !== 2)
            throw new Error("Invalid strides");
          let strideHeight = strides[0];
          let strideWidth = strides[1];
          console.log(`  strides: [${strides}]`);

          // Filter
          let filter = [];
          let global_pooling = 0;
          let filterTmp = this._getAttributeValue(args, "kernel");
          if (args.hasOwnProperty("global_pooling")) {
            global_pooling = this._getAttributeValue(args, "global_pooling");
          }
          if (typeof filterTmp.length == "undefined") {
            if (global_pooling == 1 && filterTmp == 0) {
              filter = [inputDime[1], inputDime[2]];
            } else {
              filter = [filterTmp, filterTmp];
            }
          } else {
            filter = filterTmp;
          }

          if (!filter || filter.length !== 2)
            throw new Error("Invalid filter");
          let filterWidth = filter[0];
          let filterHeight = filter[1];
          console.log(`  filter: [${filter}]`);

          // Fuse Relu
          let boundMax = 0;
          let boundMin = 0;
          if (args.hasOwnProperty("bound_max") && args.hasOwnProperty("bound_min")) {
            boundMax = this._getAttributeValue(args, "bound_max");
            boundMin = this._getAttributeValue(args, "bound_min");
            console.log(`  bound: [${boundMax}, ${boundMin}]`);
          }
          let fuseCode = this._getFuseCode(boundMax, boundMin);
          console.log(`  fuseCode: ${fuseCode}`);

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addArgInt32([paddingLeft]));
          inputs.push(this._addArgInt32([paddingRight]));
          inputs.push(this._addArgInt32([paddingTop]));
          inputs.push(this._addArgInt32([paddingBottom]));
          inputs.push(this._addArgInt32([strideWidth]));
          inputs.push(this._addArgInt32([strideHeight]));
          inputs.push(this._addArgInt32([filterWidth]));
          inputs.push(this._addArgInt32([filterHeight]));
          inputs.push(this._addArgInt32([fuseCode]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = inputTypeCode;
          let outputDims = [
            inputDime[0],
            Math.floor((inputDime[1] - filterHeight + paddingBottom + paddingTop) / strideHeight + 1),
            Math.floor((inputDime[2] - filterWidth + paddingRight + paddingLeft) / strideWidth + 1),
            inputDime[3]
          ];
          let outputType = [];
          if (this._isQuantized) {
            let outputScales = inputScales;
            if (args.hasOwnProperty("Y_scale")) {
              outputScales = this._getAttributeValue(args, "Y_scale");
            }
            let outputPoint = inputPoint;
            if (args.hasOwnProperty("Y_zero_point")) {
              outputPoint = this._getAttributeValue(args, "Y_zero_point");
            }
            outputType = {
              type: outputTypeCode,
              dimensions: outputDims,
              scale: outputScales,
              zeroPoint: outputPoint
            };
          } else {
            outputType = {
              type: outputTypeCode,
              dimensions: outputDims
            };
          }

          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          if (node.operator === "MaxPool") {
            opCode = this._nn.MAX_POOL_2D;
          } else if (node.operator === "AveragePool" || node.operator === "Int8AveragePool") {
            opCode = this._nn.AVERAGE_POOL_2D;
          }
        } break;
        case "Concat": {
          // Add inputs
          let inputTensors = node.input;
          let args = node.arg;

          // Input
          let input0Name = this._getAttributeName(inputTensors[0]);
          let input0Type = this._getTensorTypeByName(input0Name);
          let input0Dime = input0Type.dimensions;
          let input0TypeCode = input0Type.type;

          for (let i = 0; i < inputTensors.length; ++i) {
            let inputName1 = this._getAttributeName(inputTensors[i]);
            let inputType1 = this._getTensorTypeByName(inputName1);
            let inputDime1 = inputType1.dimensions;
            inputs.push(this._getTensorIdByName(inputName1));
            console.log(`  input shape: [${inputDime1}]`);
          }

          // Axis
          let axis = 3;  // default: channel(NHWC)
          if (args.hasOwnProperty("axis")) {
            axis = this._getAttributeValue(args, "axis");
            let order = this._getAttributeValue(args, "order");
            if (input0Dime.length === 4 && order == "NCHW") {
              axis = {
                0: 0,
                1: 3,
                2: 1,
                3: 2,
              }[axis];
            }
          }
          if (axis && (axis > 3 || axis < 0)) {
            throw new Error(`Invalid axis ${axis}`);
          }
          console.log(`  axis: [${axis}]`);

          inputs.push(this._addArgInt32([axis]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = input0TypeCode;
          let outputDims = [...input0Dime]
          for (let i = 1; i < inputTensors.length; ++i) {
            let inputName2 = this._getAttributeName(inputTensors[i]);
            let inputType2 = this._getTensorTypeByName(inputName2);
            outputDims[axis] += inputType2.dimensions[axis];
          }
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims
          };

          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output shape: [${outputDims}]`);

          // Add operation
          opCode = this._nn.CONCATENATION;
        } break;
        case "Dropout": {
          // Skip Dropout
          let inputTensor = node.input[0];
          let inputName = this._getAttributeName(inputTensor);
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          console.log(`Skip Dropout: ${inputName} -> ${outputName}`);
          this._tensorIds[outputName] = this._tensorIds[inputName];
          continue;
        } break;
        case "Softmax": {
          // Add inputs
          let inputTensor = node.input[0];
          let args = node.arg;

          // Input
          let inputName = this._getAttributeName(inputTensor);
          let inputType = this._getTensorTypeByName(inputName);
          let inputDime = inputType.dimensions;
          let inputTypeCode = inputType.type;

          // 4D -> 2D for dnnl, axis = 0
          if (inputDime.length == 4 && this._backend == "WebML") {
            if (inputDime.reduce((a, b) => a * b) == inputDime[3]) {
              console.log("Add reshape op for input dimensions (4D -> 2D)");
              console.log(`layer${nodeIdx}: reshape ()`);

              // Input
              let reshapeInputName = inputName;
              let reshapeDime = [1, inputDime[3]];
              let reshapeInputTypeCode = inputTypeCode;
              console.log(`  input shape: [${inputDime}]`);

              inputs.push(this._getTensorIdByName(reshapeInputName));
              inputs.push(this._addTensorInt32([2], reshapeDime));

              // Output
              let reshapeOutputName = "reshape_" + reshapeInputName;
              let reshapeOutputType = {
                type: reshapeInputTypeCode,
                dimensions: reshapeDime
              };

              let reshapeOutputID = this._addTensor(reshapeOutputName, reshapeOutputType);
              outputs.push(reshapeOutputID);
              console.log(`  output type: [${reshapeDime}]`);

              // Add operation
              opCode = this._nn.RESHAPE;

              this._addOperation(opCode, inputs, outputs);

              // Reset
              nodeIdx++;
              inputs = [];
              outputs = [];
              inputName = reshapeOutputName;
              inputType = reshapeOutputType;
              inputDime = reshapeOutputType.dimensions;
              inputTypeCode = reshapeOutputType.type;
              console.log(`layer${nodeIdx}: ${node.operator} (${node.name})`);
            }
          }

          console.log(`  input shape: [${inputDime}]`);

          // Beta
          let beta = 1.0;
          console.log(`  Beta: [${beta}]`);

          inputs.push(this._getTensorIdByName(inputName));
          inputs.push(this._addArgFloat32([beta]));

          // Add outputs
          let outputTensor = node.output[0];
          let outputName = this._getAttributeName(outputTensor);
          let outputTypeCode = inputTypeCode;
          let outputDims = inputDime;
          let outputType = {
            type: outputTypeCode,
            dimensions: outputDims
          };

          let outputID = this._addTensor(outputName, outputType);
          outputs.push(outputID);
          console.log(`  output type: [${outputDims}]`);

          // Add operation
          opCode = this._nn.SOFTMAX;
        } break;
        default: {
          throw new Error(`${node.operator} is not supported.`);
        }
      }

      this._addOperation(opCode, inputs, outputs);
    }

    // Write back all cached operands and operations
    for (let type of this._tensorTypes) {
      this._model.addOperand(type);
    }

    for (let [index, value] of Object.entries(this._operands)) {
      this._model.setOperandValue(index, value);
    }

    for (let [index, type] of Object.entries(this._quantParams)) {
      this._model.setOperandSymmPerChannelQuantParams(index, type);
    }

    for (let [opCode, inputs, outputs] of this._operations) {
      this._model.addOperation(opCode, inputs, outputs);
    }
  }

  getRequiredOps() {
    return this._requiredOps;
  }
}
