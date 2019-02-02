class PoseNet {
  constructor(modelArch, version, useAtrousConv, outputStride, inputShape, type, cacheMap, backend, prefer) {
    this._modelArch = modelArch;
    this._model = null;
    this._compilation;
    this._execution;
    this._tensorIds = [];
    this._operandIndex = 0;
    this._version = version;
    this._useAtrousConv = useAtrousConv;
    this._outputStride = outputStride;
    this._inputShape = inputShape;
    this._outputLayer = [];
    this._type = type;
    this._inputTensorId;
    this._outputTensorId;
    this._cacheMap = cacheMap;
    this._backend = backend;
    this._prefer = prefer;
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
    await this._addTensorOperands();
    await this._model.finish();
    this._compilation = await this._model.createCompilation();

    let start = performance.now();
    this._compilation.setPreference(getPreferCode(this._backend, this._prefer));
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
    let elapsed = performance.now() - start;
    console.log(`compilation time: ${elapsed.toFixed(2)} ms`);
  }

  async computeSinglePose(inputTensor, heatmapTensor, offsetTensor) {
    this._execution.setInput(0, inputTensor);
    this._execution.setOutput(0, heatmapTensor);
    this._execution.setOutput(1, offsetTensor);
    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  async computeMultiPose(inputTensor, heatmapTensor, offsetTensor, displacementFwd, displacementBwd) {
    this._execution.setInput(0, inputTensor);
    this._execution.setOutput(0, heatmapTensor);
    this._execution.setOutput(1, offsetTensor);
    this._execution.setOutput(2, displacementFwd);
    this._execution.setOutput(3, displacementBwd);
    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  async _addTensorOperands() {
    /**
    * Set model input and output layer
    * Output: 
    * Single Person: heatmap, offset
    * Multi Person: heatmap, offset, forward displacement, backward displacement
    */ 
    this._modelArch = toOutputStridedLayers(this._modelArch, this._outputStride);
    let dimensionOut;
    let dimensionIn = this._inputShape;
    const type = this._nn.TENSOR_FLOAT32;
    const heatmapDimension = [1, (this._inputShape[1]-1)/this._modelArch[13].outputStride+1, (this._inputShape[2]-1)/this._modelArch[13].outputStride+1, 17];
    const heatmap = {type: type, dimensions: heatmapDimension};
    const offsetDimension = [1, (this._inputShape[1]-1)/this._modelArch[13].outputStride+1, (this._inputShape[2]-1)/this._modelArch[13].outputStride+1, 34];
    const offset = {type: type, dimensions: offsetDimension};
    let input = {type:type, dimensions: this._inputShape};
    this._inputTensorId = this._operandIndex++;
    this._model.addOperand(input);
    this._outputTensorId = [];
    this._outputTensorId.push(this._operandIndex++);
    this._model.addOperand(heatmap);
    this._outputTensorId.push(this._operandIndex++);
    this._model.addOperand(offset);
    if (this._type === "Multiperson") {
      const displacement_fwd_dimension = [1, (this._inputShape[1]-1)/this._modelArch[13].outputStride+1, (this._inputShape[2]-1)/this._modelArch[13].outputStride+1, 32];
      const displacement_bwd_dimension = [1, (this._inputShape[1]-1)/this._modelArch[13].outputStride+1, (this._inputShape[2]-1)/this._modelArch[13].outputStride+1, 32];
      const displacement_fwd = {type:type, dimensions:displacement_fwd_dimension};
      const displacement_bwd = {type:type, dimensions:displacement_bwd_dimension};
      this._outputTensorId.push(this._operandIndex++);
      this._model.addOperand(displacement_fwd);
      this._outputTensorId.push(this._operandIndex++);
      this._model.addOperand(displacement_bwd);
    }
    this._model.identifyInputsAndOutputs([this._inputTensorId], this._outputTensorId);

    let outputLayerIndex = 0; 
    let manifest = await fetchDataByUrl(getURL(this._version)+"manifest.json", false);
    manifest = JSON.parse(manifest);
    for (let i in this._modelArch) {
      this._calculateProgress(Number(i)+1, this._modelArch.length);
      let dimensionWeights = [];
      let weights = [];
      let dimensionBias = [];
      let bias = [];
      let inputs = [];
      if (this._modelArch[i]["convType"] === "conv2d") {
        if (i == 0) {
          inputs.push(this._inputTensorId);
        } else {
          inputs.push(this._outputLayer[outputLayerIndex]);
          outputLayerIndex++;
        }	  
        /** 
         * data = { 
         *          shapeWeights,
         *          weights,
         *          shapeBias,
         *          bias
         *        }
         */
        const data = await getDimensionData("conv2d", this._version, i, manifest, this._cacheMap);
        dimensionWeights.push(reshape(data.shapeWeights));
        weights.push(new Float32Array(transposeWeights(data.weights, data.shapeWeights)));
        dimensionBias.push(data.shapeBias);
        bias.push(data.bias);
        dimensionOut = this._calculateOutput(dimensionIn, dimensionWeights[0],
                                             this._modelArch[i]["stride"], "conv2d");
        dimensionIn = dimensionOut;
        this._outputLayer.push(this._operandIndex);
        this._model.addOperand({type: type, dimensions: dimensionOut});
        this._operandIndex++;
      }
      // separableConv = [Depthwise Convolution, Pointwise Convolution]
      if (this._modelArch[i]["convType"] === "separableConv") {
        if (i == 0) {
          inputs.push(this._inputTensorId);
        } else {
          inputs.push(this._outputLayer[outputLayerIndex]);
          outputLayerIndex++;
        }
        const data = await getDimensionData("separableConv", this._version, i, manifest, this._cacheMap);
        if (this._useAtrousConv || this._modelArch[i].rate === 1) { 
          dimensionWeights.push(reshape(data.shapeWeights[0]));
          dimensionWeights.push(reshape(data.shapeWeights[1]));
          weights.push(new Float32Array(transposeWeights(data.weights[0], data.shapeWeights[0])));
          weights.push(new Float32Array(transposeWeights(data.weights[1], data.shapeWeights[1])));
          dimensionBias.push(data.shapeBias[0]);
          dimensionBias.push(data.shapeBias[1]);
          bias.push(data.bias[0]);
          bias.push(data.bias[1]);
          dimensionOut = this._calculateOutput(dimensionIn, dimensionWeights[0], 
                                               this._modelArch[i]["stride"], "depthwise");
        }
        else {
          const dilationData = dilationWeights(new Float32Array(transposeWeights(data.weights[0], data.shapeWeights[0])),
                                               reshape(data.shapeWeights[0]), this._modelArch[i].rate);
          dimensionWeights.push(dilationData.dimension);
          dimensionWeights.push(reshape(data.shapeWeights[1]));
          weights.push(dilationData.dilationWeights);
          weights.push(new Float32Array(transposeWeights(data.weights[1], data.shapeWeights[1])));
          dimensionBias.push(data.shapeBias[0]);
          dimensionBias.push(data.shapeBias[1]);
          bias.push(data.bias[0]);
          bias.push(data.bias[1]);
          dimensionOut = this._calculateOutput(dimensionIn, reshape(data.shapeWeights[0]), 
                                               this._modelArch[i]["stride"], "depthwise");
        }
        dimensionIn = dimensionOut;
        this._outputLayer.push(this._operandIndex);
        this._model.addOperand({type:type, dimensions: dimensionOut});
        this._operandIndex++;
        dimensionOut = this._calculateOutput(dimensionIn, dimensionWeights[1], 1, "pointwise");
        dimensionIn = dimensionOut;
        this._outputLayer.push(this._operandIndex);
        this._model.addOperand({type:type, dimensions: dimensionOut});
        this._operandIndex++;
      }
      for (let j in dimensionWeights) {
        let tensorId = this._operandIndex++;
        let tensorTypeWeights = {type: type, dimensions: dimensionWeights[j]};
        this._model.addOperand(tensorTypeWeights);
        this._model.setOperandValue(tensorId, weights[j]);
        inputs.push(tensorId);
        tensorId = this._operandIndex++;
        let tensorTypeBias = {type: type, dimensions: dimensionBias[j]};
        this._model.addOperand(tensorTypeBias);
        this._model.setOperandValue(tensorId, bias[j]);
        inputs.push(tensorId);
        if (this._modelArch[i].convType === "conv2d") {
          let outputs = this._outputLayer[outputLayerIndex];
          const paddingCode = this._nn.PADDING_SAME;
          const fuseCode = this._nn.FUSED_RELU6;
          inputs.push(this._addScalarInt32(paddingCode));
          inputs.push(this._addScalarInt32(this._modelArch[i].stride));
          inputs.push(this._addScalarInt32(this._modelArch[i].stride));
          inputs.push(this._addScalarInt32(fuseCode));
          const opType = this._nn.CONV_2D;
          this._model.addOperation(opType, inputs, [outputs]);
        } else if (this._modelArch[i].convType === "separableConv") {
          const paddingCode = this._nn.PADDING_SAME;
          const fuseCode = this._nn.FUSED_RELU6;
          if (j == 0) {
            let opType;
            const multiplier = 1;
            let outputs = this._outputLayer[outputLayerIndex];
            inputs.push(this._addScalarInt32(paddingCode));
            if (this._useAtrousConv && this._modelArch[i].rate !== 1) {
              inputs.push(this._addScalarInt32(this._modelArch[i].rate));
              inputs.push(this._addScalarInt32(this._modelArch[i].rate));
              opType = this._nn.ATROUS_DEPTHWISE_CONV_2D;
            } else {
              inputs.push(this._addScalarInt32(this._modelArch[i].stride));
              inputs.push(this._addScalarInt32(this._modelArch[i].stride));
              opType = this._nn.DEPTHWISE_CONV_2D;
            }
            inputs.push(this._addScalarInt32(multiplier));
            inputs.push(this._addScalarInt32(fuseCode));
            this._model.addOperation(opType, inputs, [outputs]);
            inputs = [];
          } else {
            const stride = 1;
            inputs.unshift(this._outputLayer[outputLayerIndex]);
            outputLayerIndex++;
            const opType = this._nn.CONV_2D;
            let outputs = this._outputLayer[outputLayerIndex];
            inputs.push(this._addScalarInt32(paddingCode));
            inputs.push(this._addScalarInt32(stride));
            inputs.push(this._addScalarInt32(stride));
            inputs.push(this._addScalarInt32(fuseCode));
            this._model.addOperation(opType, inputs, [outputs]);
          }
        }
      }    
    }

    // add operation and operands for output layer
    let tensorTypeWeights, tensorTypeBias;
    let valueWeights, valueBias;
    let inputs = [];
    const stride = 1;
    inputs.push(this._outputLayer[this._outputLayer.length-1]);
    let data = await getDimensionData("heatmap", this._version, 0, manifest, this._cacheMap);
    tensorTypeWeights = {type: type, dimensions: reshape(data.shapeWeights)};
    tensorTypeBias = {type: type, dimensions: data.shapeBias};
    valueWeights = new Float32Array(transposeWeights(data.weights, data.shapeWeights));
    valueBias = data.bias;
    let tensorId = this._operandIndex++;
    this._model.addOperand(tensorTypeWeights);
    this._model.setOperandValue(tensorId, valueWeights);
    inputs.push(tensorId);
    tensorId = this._operandIndex++;
    this._model.addOperand(tensorTypeBias);
    this._model.setOperandValue(tensorId, valueBias);
    inputs.push(tensorId);
    inputs.push(this._addScalarInt32(this._nn.PADDING_SAME));
    inputs.push(this._addScalarInt32(stride));
    inputs.push(this._addScalarInt32(stride));
    inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
    let outputs = this._outputTensorId[0];
    this._model.addOperation(this._nn.CONV_2D, inputs, [outputs]);

    // add operands for offset layer
    inputs = [];
    inputs.push(this._outputLayer[this._outputLayer.length-1]);
    data = await getDimensionData("offset", this._version, 0, manifest, this._cacheMap)
    tensorTypeWeights = {type: type, dimensions: reshape(data.shapeWeights)};
    tensorTypeBias = {type: type, dimensions: data.shapeBias};
    valueWeights = new Float32Array(transposeWeights(data.weights, data.shapeWeights));
    valueBias = data.bias;

    tensorId = this._operandIndex++;
    this._model.addOperand(tensorTypeWeights);
    this._model.setOperandValue(tensorId, valueWeights);
    inputs.push(tensorId);
    tensorId = this._operandIndex++;
    this._model.addOperand(tensorTypeBias);
    this._model.setOperandValue(tensorId, valueBias);
    inputs.push(tensorId);
    inputs.push(this._addScalarInt32(this._nn.PADDING_SAME));
    inputs.push(this._addScalarInt32(stride));
    inputs.push(this._addScalarInt32(stride));
    inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
    outputs = this._outputTensorId[1];
    this._model.addOperation(this._nn.CONV_2D, inputs, [outputs]);
    
    if (this._type === "Multiperson") {  
      inputs = [];
      inputs.push(this._outputLayer[this._outputLayer.length-1]);  
      data = await getDimensionData("displacement_fwd", this._version, 0, manifest, this._cacheMap)
      tensorTypeWeights = {type: type, dimensions: reshape(data.shapeWeights)};
      tensorTypeBias = {type: type, dimensions: data.shapeBias};
      valueWeights = new Float32Array(transposeWeights(data.weights, data.shapeWeights));
      valueBias = data.bias;

      tensorId = this._operandIndex++;
      this._model.addOperand(tensorTypeWeights);
      this._model.setOperandValue(tensorId, valueWeights);
      inputs.push(tensorId);
      tensorId = this._operandIndex++;
      this._model.addOperand(tensorTypeBias);
      this._model.setOperandValue(tensorId, valueBias);
      inputs.push(tensorId);
      inputs.push(this._addScalarInt32(this._nn.PADDING_SAME));
      inputs.push(this._addScalarInt32(stride));
      inputs.push(this._addScalarInt32(stride));
      inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
      outputs = this._outputTensorId[2];
      this._model.addOperation(this._nn.CONV_2D, inputs, [outputs]);
      
      inputs = [];
      inputs.push(this._outputLayer[this._outputLayer.length-1]);  
      data = await getDimensionData("displacement_bwd", this._version, 0, manifest, this._cacheMap)
      tensorTypeWeights = {type: type, dimensions: reshape(data.shapeWeights)};
      tensorTypeBias = {type: type, dimensions: data.shapeBias};
      valueWeights = new Float32Array(transposeWeights(data.weights, data.shapeWeights));
      valueBias = data.bias;

      tensorId = this._operandIndex++;
      this._model.addOperand(tensorTypeWeights);
      this._model.setOperandValue(tensorId, valueWeights);
      inputs.push(tensorId);
      tensorId = this._operandIndex++;
      this._model.addOperand(tensorTypeBias);
      this._model.setOperandValue(tensorId, valueBias);
      inputs.push(tensorId);
      inputs.push(this._addScalarInt32(this._nn.PADDING_SAME));
      inputs.push(this._addScalarInt32(stride));
      inputs.push(this._addScalarInt32(stride));
      inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
      outputs = this._outputTensorId[3];
      this._model.addOperation(this._nn.CONV_2D, inputs, [outputs]);
    }
  }

  _calculateOutput(inputDimension, shape, stride, layer) {
    let outputDimension;
    if (layer === "conv2d") {
      outputDimension = [1, Math.floor((inputDimension[1]-shape[1]+2)/stride+1), 
                         Math.floor((inputDimension[2]-shape[2]+2)/stride+1), shape[0]];
    } else if (layer === "depthwise") {
      outputDimension = [1, Math.floor((inputDimension[1]-shape[1]+2)/stride+1), 
                         Math.floor((inputDimension[2]-shape[2]+2)/stride+1), inputDimension[3]];
    } else {
      outputDimension = [1, Math.floor((inputDimension[1]-shape[1]+0)/stride+1), 
                         Math.floor((inputDimension[2]-shape[2]+0)/stride+1), shape[0]];
    }
    return outputDimension;
  }


  _addScalarInt32(value) {
    const scalarInt32Type = {type: this._nn.INT32};
    let index = this._operandIndex++;
    this._model.addOperand(scalarInt32Type);
    this._model.setOperandValue(index, new Int32Array([value]));
    return index;
  }

  _calculateProgress(current, length) {
    let progressBar = document.getElementById('progressBar');
    let progressContainer = document.getElementById('progressContainer');
    if (progressBar !== null && progressContainer !== null) {
      let totalSize = length;
      let loadedSize = current;
      let percentComplete = current / length *100;
      percentComplete = percentComplete.toFixed(0);
      progressBar.style = `width: ${percentComplete}%`;
      updateLoading(loadedSize.toFixed(0), totalSize.toFixed(0), percentComplete);
    }
  }
}


