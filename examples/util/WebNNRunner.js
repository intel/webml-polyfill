class WebNNRunner extends BaseRunner {
  constructor() {
    super();
    this._currentBackend = null;
    this._currentPrefer = null;
    this._outputTensor = [];
    this._rawModel = null;
    this._subgraphsSummary = [];
    this._modelRequiredOps = null;
    this._deQuantizeParams = null;
    this._bEagerMode = false;
    this._supportedOps = new Set();
  }

  /**
   * This method is to set '_currentBackend'.
   * @param backend: A string for backend.
   */
  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  /**
   * This method is to set '_currentPrefer'.
   * @param prefer: A string for prefer.
   */
  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  /**
   * This method is to set '_bEagerMode'.
   * @param flag: A boolean whether eager mode.
   */
  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  /**
   * This method is to set '_supportedOps'.
   * @param ops: A string for supported ops.
   */
  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  /**
   * This method is to set '_rawModel'.
   * @param model: A string for model.
   */
  _setRawModel = (model) => {
    this._rawModel = model;
  };

  /**
   * This method is to set '_subgraphsSummary'.
   * @param summary: A string for summary.
   */
  _setSubgraphsSummary = (summary) => {
    this._subgraphsSummary = summary;
  };

  /**
   * This method is to set '_modelRequiredOps'.
   * @param ops: A string for required ops of model.
   */
  _setModelRequiredOps = (ops) => {
    this._modelRequiredOps = ops;
  };

  /**
   * This method is to set '_deQuantizeParams'.
   * @param params: A string for deQuantize params.
   */
  _setDeQuantizeParams = (params) => {
    this._deQuantizeParams = params;
  };

  /**
   * This method is to load model file with specified url.
   * @param url: A string for model file url.
   * @returns {string} This returns status of load model file, ['ERROR' | 'SUCCESS'].
   */
  _loadModelFile = async (url) => {
    let status = 'ERROR';
    let rawModel = null;

    if (url !== undefined) {
      const arrayBuffer = await this._loadURL(url, this._progressHandler, true);
      const bytes = new Uint8Array(arrayBuffer);
      switch (url.split('.').pop()) {
        case 'tflite':
          const flatBuffer = new flatbuffers.ByteBuffer(bytes);
          rawModel = tflite.Model.getRootAsModel(flatBuffer);
          rawModel._rawFormat = 'TFLITE';
          status = 'SUCCESS'
          printTfLiteModel(rawModel);
          break;
        case 'onnx':
          const err = onnx.ModelProto.verify(bytes);
          if (err) {
            throw new Error(`The model file ${url} is invalid, ${err}`);
          }
          rawModel = onnx.ModelProto.decode(bytes);
          rawModel._rawFormat = 'ONNX';
          status = 'SUCCESS'
          printOnnxModel(rawModel);
          break;
        case 'bin':
          const networkFile = url.replace(/bin$/, 'xml');
          const networkText = await this._loadURL(networkFile);
          const weightsBuffer = bytes.buffer;
          rawModel = new OpenVINOModel(networkText, weightsBuffer);
          rawModel._rawFormat = 'OPENVINO';
          status = 'SUCCESS';
          break;
        default:
          throw new Error(`Unrecognized model format, support TFLite | ONNX | OpenVINO model`);
      }
    } else {
      throw new Error(`There's none model file info, please check config info of modelZoo.`);
    }

    this._setRawModel(rawModel);
    this._setLoadedFlag(true);
    return status;
  };

  /**
   * This method is to load model with given model info.
   * @param modelInfo: A string for model info.
   * @returns {string} This returns 'LOADED' status if model already loaded.
   */
  loadModel = async (modelInfo) => {
    if (this._bLoaded && this._currentModelInfo.modelFile === modelInfo.modelFile) {
      return 'LOADED';
    }

    // reset all states
    this._setLoadedFlag(false);
    this._setInitializedFlag(false);
    this._setBackend(null);
    this._setPrefer(null);
    this._setModelInfo(modelInfo);
    this._setModelRequiredOps(new Set());
    this._setDeQuantizeParams([]);
    this._setSubgraphsSummary([]);
    this._initInputTensor();
    this._initOutputTensor();

    await this._loadModelFile(this._currentModelInfo.modelFile);

    if (this._currentModelInfo.labelsFile != null) {
      await this._loadLabelsFile(this._currentModelInfo.labelsFile);
    }
  };

  /**
   * This method is to get typedArray type for inputTensor.
   * @returns {object} This returns Uint8Array or Float32Array object or other typedArray type if inherited.
   */
  _getInputTensorTypedArray = () => {
    // Override by inherited if needed
    const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    return typedArray;
  };

  /**
   * This method is to init set '_inputTensor'.
   */
  _initInputTensor = () => {
    const typedArray = this._getInputTensorTypedArray();
    this._inputTensor = [new typedArray(this._currentModelInfo.inputSize.reduce((a, b) => a * b))];
  };

  /**
   * This method is to typedArray type for outputTensor.
   * @returns {object} This returns Uint8Array or Float32Array object or other typedArray type if inherited.
   */
  _getOutputTensorTypedArray = () => {
    // Override by inherited if needed
    const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
    return typedArray;
  };

  /**
   * This method is to init set '_outputTensor'.
   */
  _initOutputTensor = () => {
    // Override by inherited if needed
    const typedArray = this._getOutputTensorTypedArray();
    const outputSize = this._currentModelInfo.outputSize;

    if (typeof outputSize === 'number') {
      this._outputTensor = [new typedArray(outputSize)];
    } else {
      this._outputTensor = [new typedArray(outputSize.reduce((a, b) => a * b))];
    }
  };

  /**
   * This method is to compile a machine learning model by backend and prefer.
   * @param options: A string has backend and prefer info.
   * @returns {string} This returns a string for status of compilation model.
   */
  compileModel = async (options) => {
    const backend = options.backend;
    const prefer = options.prefer;
    if (!this._bLoaded) {
      return 'NOT_LOADED';
    }

    if (this._bInitialized && backend === this._currentBackend && prefer === this._currentPrefer) {
      return 'INITIALIZED';
    }

    this._setBackend(backend);
    this._setPrefer(prefer);
    this._setInitializedFlag(false);
    const postOptions = this._currentModelInfo.postOptions || {};
    const configs = {
      rawModel: this._rawModel,
      backend: this._currentBackend,
      prefer: this._currentPrefer,
      softmax: postOptions.softmax || false,
    };

    let model = null;

    switch (this._rawModel._rawFormat) {
      case 'TFLITE':
        model = new TFliteModelImporter(configs);
        break;
      case 'ONNX':
        model = new OnnxModelImporter(configs);
        break;
      case 'OPENVINO':
        model = new OpenVINOModelImporter(configs);
        break;
      default:
        throw new Error(`Unsupported '${rawModel._rawFormat}' model.`);
    }

    this._setModel(model);
    this._model.setSupportedOps(this._supportedOps);
    this._model.setEagerMode(this._bEagerMode);
    await this._model.createCompiledModel();

    this._setModelRequiredOps(this._model.getRequiredOps());

    if (this._currentModelInfo.isQuantized) {
      this._setDeQuantizeParams(model._deQuantizeParams);
    }

    if (this._currentBackend !== 'WebML' && model._compilation && model._compilation._preparedModel) {
       this._setSubgraphsSummary(model._compilation._preparedModel.getSubgraphsSummary());
    }

    // Warm up model
    const computeStart = performance.now();
    const computeStatus = await this._model.compute(this._inputTensor, this._outputTensor);
    const computeDelta = performance.now() - computeStart;
    console.log(`Computed Status: [${computeStatus}]`);
    console.log(`Warm up Time: ${computeDelta.toFixed(2)} ms`);

    this._setInitializedFlag(true);
    return 'SUCCESS';
  };

  /**
   * This method is to do inference with input src (HTML [<img> | <video> | <audio>] element).
   * @param src: An object for HTML [<img> | <video> | <audio>] element.
   * @param options: A string to get inputTensor. Details:
   * options = {
   *   // inputSize was configed in modelZoo.js, inputSize = [h, w, c] or [1, size] for audio example.
   *   inputSize: inputSize,
   *   // preOptions was also configed in modelZoo.js,
   *   // preOptions= {} or {mean: [number, number, number, number],std: [number, number, number, number]}
   *   preOptions: preOptions,
   *   imageChannels: 4,
   *   drawOptions: { // optional, drawOptions is used for CanvasRenderingContext2D.drawImage() method.
   *     sx: sx, // the x-axis coordinate of the top left corner of sub-retangle of the source image
   *     sy: sy, // the y-axis coordinate of the top left corner of sub-retangle of the source image
   *     sWidth: sWidth, // the width of the sub-retangle of the source image
   *     sHeight: sHeight, // the height of the sub-retangle of the source image
   *     dWidth: dWidth, // the width to draw the image in the detination canvas
   *     dHeight: dWidth, // the height to draw the image in the detination canvas
   *   },
   *   scaledFlag: true, // optional, need scaled the width and height of element to get need inputTensor
   * };
   * @returns {string} This returns a string for inference status.
   */
  run = async (src, options) => {
    let status = 'ERROR';

    if (src.tagName === 'AUDIO') {
      await getTensorArrayByAudio(src, this._inputTensor, options);
    } else {
      getTensorArray(src, this._inputTensor, options);
    }

    const start = performance.now();
    status = await this._model.compute(this._inputTensor, this._outputTensor);
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${delta} ms]`);
    return status;
  };

  /**
   * This method is get required ops of model.
   * @returns {object} This returns an array object for required ops of model.
   */
  getRequiredOps = () => {
    return this._modelRequiredOps;
  };

  /**
   * This method is to get inference subgraphs summary info.
   * @returns {object} This returns an array object for inference subgraphs summary info.
   */
  getSubgraphsSummary = () => {
    return this._subgraphsSummary;
  };

  /**
   * This method is to get deQuantize params of deQuantized model.
   * @returns {object} This returns an array object for deQuantize params of deQuantized model.
   */
  getDeQuantizeParams = () => {
    return this._deQuantizeParams;
  };

  /**
   * This method is to output inference tensor for post processing by example side.
   * @param output: An object for output which will be updated with output tensor info by this method.
   */
  _passOutputTensor = (output) => {
    // Override by inherited if needed
    output.outputTensor = this._outputTensor[0];
  };

  /**
   * This method is to free allocated memory resources for model compilation process by polyfill backend.
   */
  deleteAll = () => {
    if (this._currentBackend != 'WebML') {
      // free allocated memory on compilation process by polyfill WASM / WebGL backend.
      if (this._model._compilation && this._model._compilation._preparedModel) {
        this._model._compilation._preparedModel._deleteAll();
      }
    }
  };

  /**
   * This method is for debugging output of each layer after ran the example once on Console, usage likes:
   *   example._runner.iterateLayers([{backend: 'WASM', prefer: 'fast'}], [1, 2]) // debugging the outputs of first and second layers using WASM backend
   *   example._runner.iterateLayers([{backend: 'WASM', prefer: 'fast'}]) // debugging outputs of all layers using WASM backend
   *   example._runner.iterateLayers([{backend: 'WASM', prefer: 'fast'}, {backend: 'WebML', prefer: 'fast'}], [1, 2]) // debugging outputs of all layers using WASM backend and WebML backend, user can compare each output of the same layer.
   *   example._coRunner.iterateLayers([{backend: 'WASM', prefer: 'fast'}, {backend: 'WebML', prefer: 'fast'}]) if wanted debugging cowork model for those examples with cowork models.
   * The debugging output info logs are printed on Console with enabling verbose level.
   * @param configs: An object for config array, likes:
   *   [{backend: 'WASM', prefer: 'fast'}]
   *   [{backend: 'WebGL', prefer: 'sustained'}]
   *   [{backend: 'WebML', prefer: 'fast'}]
   *   [{backend: 'WebML', prefer: 'sustained'}]
   *   or
   *   [{backend: 'WASM', prefer: 'fast'}, {backend: 'WebGL', prefer: 'sustained'}]
   *   [{backend: 'WASM', prefer: 'fast'}, {backend: 'WebML', prefer: 'fast'}]
   *   etc.
   * @param layerList: {object | undefined} An object for layer array, likes: [1, 2 ,3], if undefined, it means all layers.
   */
  iterateLayers = async (configs, layerList) => {
    if (!this._bInitialized) return;

    const iterators = [];
    const models = [];

    for (const config of configs) {
      const fileExtension = this._currentModelInfo.modelFile.split('.').pop();
      const importer = {
        tflite: TFliteModelImporter,
        onnx: OnnxModelImporter,
        bin: OpenVINOModelImporter,
      }[fileExtension];
      const model = await new importer({
        isQuantized: this._currentModelInfo.isQuantized,
        rawModel: this._rawModel,
        backend: config.backend,
        prefer: config.prefer || null,
      });
      iterators.push(model.layerIterator(this._inputTensor, layerList));
      models.push(model);
    }

    while (true) {
      let layerOutputs = [];
      for (let it of iterators) {
        layerOutputs.push(await it.next());
      }
      let refOutput = layerOutputs[0];
      if (refOutput.done) {
        break;
      }
      console.debug(`\n\n\nLayer(${refOutput.value.layerId}) ${refOutput.value.outputName}`);
      for (let i = 0; i < configs.length; ++i) {
        console.debug(`\n${configs[i].backend}:`);
        console.debug(`\n${layerOutputs[i].value.tensor}`);
        if (i > 0) {
          let sum = 0;
          for (let j = 0; j < refOutput.value.tensor.length; j++) {
            sum += Math.pow(layerOutputs[i].value.tensor[j] - refOutput.value.tensor[j], 2);
          }
          let variance = sum / refOutput.value.tensor.length;
          console.debug(`var with ${configs[0].backend}: ${variance}`);
        }
      }
    }

    for (let model of models) {
      if (model._backend !== 'WebML') {
        model._compilation._preparedModel._deleteAll();
      }
    }
  };
}
