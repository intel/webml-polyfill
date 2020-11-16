class WebNNRunner extends BaseRunner {
  constructor() {
    super();
    this._currentBackend = null;
    this._currentPrefer = null;
    this._inputTensor = [];
    this._outputTensor = [];
    this._rawModel = null;
    this._subgraphsSummary = [];
    this._modelRequiredOps = null;
    this._deQuantizeParams = null;
    this._bEagerMode = false;
    this._supportedOps = [];
  }

  /**
   * This method is to set '_currentBackend'.
   * @param {string} backend
   */
  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  /**
   * This method is to set '_currentPrefer'.
   * @param {string} prefer
   */
  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  /**
   * This method is to set '_bEagerMode'.
   * @param {boolean} flag
   */
  _setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  /**
   * This method is to set '_supportedOps'.
   * @param {!Array<number>} ops
   */
  _setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  /**
   * This method is to set '_rawModel'.
   * @param {object} model
   */
  _setRawModel = (model) => {
    this._rawModel = model;
  };

  /**
   * This method is to set '_subgraphsSummary'.
   * @param {object} summary An array object that for summary.
   */
  _setSubgraphsSummary = (summary) => {
    this._subgraphsSummary = summary;
  };

  /**
   * This method is to set '_modelRequiredOps'.
   * @param {!Array<number>} ops
   */
  _setModelRequiredOps = (ops) => {
    this._modelRequiredOps = ops;
  };

  /**
   * This method is to set '_deQuantizeParams'.
   * @param {object} params An array object that for deQuantize params.
   */
  _setDeQuantizeParams = (params) => {
    this._deQuantizeParams = params;
  };

  /** @override */
  _loadModelFile = async (url) => {
    let rawModel = null;

    if (url !== undefined) {
      const arrayBuffer = await this._loadURL(url, this._progressHandler, true);
      const bytes = new Uint8Array(arrayBuffer);
      switch (url.split('.').pop()) {
        case 'tflite':
          const flatBuffer = new flatbuffers.ByteBuffer(bytes);
          rawModel = tflite.Model.getRootAsModel(flatBuffer);
          rawModel._rawFormat = 'TFLITE';
          printTfLiteModel(rawModel);
          break;
        case 'onnx':
          const err = onnx.ModelProto.verify(bytes);
          if (err) {
            throw new Error(`The model file ${url} is invalid, ${err}`);
          }
          rawModel = onnx.ModelProto.decode(bytes);
          rawModel._rawFormat = 'ONNX';
          printOnnxModel(rawModel);
          break;
        case 'bin':
          const networkFile = url.replace(/bin$/, 'xml');
          const networkText = await this._loadURL(networkFile);
          const weightsBuffer = bytes.buffer;
          rawModel = new OpenVINOModel(networkText, weightsBuffer);
          rawModel._rawFormat = 'OPENVINO';
          break;
        case 'pb':
          const weightFile = url.replace(/predict/, 'init');
          const weightBuffer = await this._loadURL(weightFile, this._progressHandler, true);
          const weightBytes = new Uint8Array(weightBuffer);
          const netBuffer = bytes;
          const weightMessage = protobuf.roots["caffe2"].caffe2.NetDef.decode(weightBytes);
          const netMessage = protobuf.roots["caffe2"].caffe2.NetDef.decode(netBuffer);
          const caffe2Utils = new Caffe2ModelUtils(netMessage,
                                                   weightMessage,
                                                   this._currentModelInfo.isDNNL);
          rawModel = [...caffe2Utils.getCaffe2Model()];
          rawModel._rawFormat = 'CAFFE2';
          break;
        default:
          throw new Error(`Unrecognized model format, support TFLite | ONNX | OpenVINO model`);
      }
    } else {
      throw new Error(`There's none model file info, please check config info of modelZoo.`);
    }

    this._setRawModel(rawModel);
    this._setLoadedFlag(true);
  };

  /**
   * This method is to get typedArray type for inputTensor.
   * @returns {function} This returns Uint8Array or Float32Array function or other typedArray function if inherited.
   */
  _getInputTensorTypedArray = () => {
    if (this._currentModelInfo.isQuantized || false) {
      if (this._currentModelInfo.isDNNL || false) {
        return Int8Array;
      } else if (this._currentModelInfo.isIE || false){
        return Float32Array;
      } else {
        return Uint8Array;
      }
    } else {
      return Float32Array;
    }
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
   * @returns {function} This returns Uint8Array or Float32Array function or other typedArray function if inherited.
   */
  _getOutputTensorTypedArray = () => {
    // Override by inherited if needed
    const typedArray = this._currentModelInfo.isQuantized || false ? (this._currentModelInfo.isDNNL || this._currentModelInfo.isIE || false ? Float32Array : Uint8Array) : Float32Array;
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

  /** @override */
  doInitialization = (modelInfo) => {
    this._setLoadedFlag(false);
    this._setInitializedFlag(false);
    this._setBackend(null);
    this._setPrefer(null);
    this._setModelRequiredOps(new Set());
    this._setDeQuantizeParams([]);
    this._setSubgraphsSummary([]);
    this._setModelInfo(modelInfo);
    this._initInputTensor();
    this._initOutputTensor();
  };

  /** @override */
  _checkInitializedCompilation = (options) => {
    return this._bInitialized
           && this._currentBackend === options.backend
           && this._currentPrefer === options.prefer
           && this._bEagerMode === options._bEagerMode
           && this._supportedOps.toString() === options.supportedOps.toString();
  }

  /** @override */
  _doCompile = async (options) => {
    let model = null;
    const backend = options.backend;
    const prefer = options.prefer;
    const eagerMode = options.eagerMode || false;
    const supportedOps = options.supportedOps || [];
    this._freeAllocatedMemory();
    this._setBackend(backend);
    this._setPrefer(prefer);
    this._setEagerMode(eagerMode);
    this._setSupportedOps(supportedOps);

    const postOptions = this._currentModelInfo.postOptions || {};
    const configs = {
      rawModel: this._rawModel,
      backend: this._currentBackend,
      prefer: this._currentPrefer,
      softmax: postOptions.softmax || false,
      inputScaleFactor: options.scaleFactor, // for GNA
      isQuantized: this._currentModelInfo.isQuantized || false,
      isIE: this._currentModelInfo.isIE || false,
      isDNNL: this._currentModelInfo.isDNNL || false,
      inputSize: this._currentModelInfo.inputSize // for caffe2 model
    };

    if (configs.backend !== 'WebML' &&
      configs.isQuantized === true && configs.isIE === true) {
      throw new Error(`This backend hasn't supported OpenVINO quantized models.`);
    }

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
      case 'CAFFE2':
        model = new Caffe2ModelImporter(configs);
        break;
      default:
        throw new Error(`Unsupported '${rawModel._rawFormat}' model.`);
    }

    this._setModel(model);
    this._model.setSupportedOps(new Set(this._supportedOps));
    this._model.setEagerMode(this._bEagerMode);
    await this._model.createCompiledModel();

    this._saveDetails();
    await this._doWarmup();
  };

  /**
   * This method is to save relevant details info of
   * 1. model's required ops,
   * 2. dequantize params if model was quantized,
   * 3. subgraphs summary info.
   */
  _saveDetails = () => {
    this._setModelRequiredOps(this._model.getRequiredOps());

    if (this._currentModelInfo.isQuantized) {
      this._setDeQuantizeParams(this._model._deQuantizeParams);
    }

    if (this._currentBackend !== 'WebML' && this._model._compilation && this._model._compilation._preparedModel) {
      this._setSubgraphsSummary(this._model._compilation._preparedModel.getSubgraphsSummary());
    }
  };

  /**
   * This method is to do warm up with compiled model.
   */
  _doWarmup = async () => {
    // Warm up model
    const computeStart = performance.now();
    const computeStatus = await this._model.compute(this._inputTensor, this._outputTensor);
    const computeDelta = performance.now() - computeStart;
    console.log(`Computed Status: [${computeStatus}]`);
    console.log(`Warm up Time: ${computeDelta.toFixed(2)} ms`);
  };

  /**
   * This method is to set '_inputTensor' with input.
   * @param {!Object<string, *>} input
   *     input = {
   *       src: !HTMLElement, //An object for HTML [<img> | <video> | <audio>] element.
   *       options: { // An object to get input tensor.
   *         // inputSize was configed in modelZoo.js, inputSize = [h, w, c] or [1, size] for audio example.
   *         inputSize: {!Array<number>},
   *         // preOptions was also configed in modelZoo.js,
   *         // preOptions= {} or likes {mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5],}
   *         preOptions: {!Object<string, *>},
   *         imageChannels: {number},
   *         drawOptions: { // optional, drawOptions is used for CanvasRenderingContext2D.drawImage() method.
   *           sx: {number}, // the x-axis coordinate of the top left corner of sub-retangle of the source image
   *           sy: {number}, // the y-axis coordinate of the top left corner of sub-retangle of the source image
   *           sWidth: {number}, // the width of the sub-retangle of the source image
   *           sHeight: {number}, // the height of the sub-retangle of the source image
   *           dWidth: {number}, // the width to draw the image in the detination canvas
   *           dHeight: {number}, // the height to draw the image in the detination canvas
   *         },
   *         scaledFlag: {boolean}, // optional, need scaled the width and height of element to get need inputTensor
   *       },
   *     };
   */
  _getTensor = (input) => {
    const image = input.src;
    const options = input.options;
    let tensor = this._inputTensor[0];

    image.width = image.videoWidth || image.naturalWidth;
    image.height = image.videoHeight || image.naturalHeight;

    const [height, width, channels] = options.inputSize;
    const preOptions = options.preOptions || {};
    const mean = preOptions.mean || [0, 0, 0, 0];
    const std = preOptions.std || [1, 1, 1, 1];
    const normlizationFlag = preOptions.norm || false;
    const channelScheme = preOptions.channelScheme || 'RGB';
    const imageChannels = options.imageChannels || 4; // RGBA
    const drawOptions = options.drawOptions;
    const nchwFlag = preOptions.nchwFlag || false;

    let canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    let canvasContext = canvasElement.getContext('2d');

    if (drawOptions) {
      canvasContext.drawImage(image, drawOptions.sx, drawOptions.sy, drawOptions.sWidth, drawOptions.sHeight,
        0, 0, drawOptions.dWidth, drawOptions.dHeight);
    } else {
      if (options.scaledFlag) {
        const resizeRatio = Math.max(Math.max(image.width / width, image.height / height), 1);
        const scaledWidth = Math.floor(image.width / resizeRatio);
        const scaledHeight = Math.floor(image.height / resizeRatio);
        canvasContext.drawImage(image, 0, 0, scaledWidth, scaledHeight);
      } else {
        canvasContext.drawImage(image, 0, 0, width, height);
      }
    }

    let pixels = canvasContext.getImageData(0, 0, width, height).data;

    if (normlizationFlag) {
      pixels = new Float32Array(pixels).map(p => p / 255);
    }

    if (channelScheme === 'RGB') {
      if (channels > 1) {
        for (let c = 0; c < channels; ++c) {
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              let value = pixels[h * width * imageChannels + w * imageChannels + c];
              if (nchwFlag) {
                tensor[c * width * height + h * width + w] = (value - mean[c]) / std[c];
              } else {
                tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
              }
            }
          }
        }

      } else if (channels === 1) {
        for (let c = 0; c < channels; ++c) {
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              let index = h * width * imageChannels + w * imageChannels + c;
              let value = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
              if (nchwFlag) {
                tensor[c * width * height + h * width + w] = (value - mean[c]) / std[c];
              } else {
                tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
              }
            }
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      for (let c = 0; c < channels; ++c) {
        for (let h = 0; h < height; ++h) {
          for (let w = 0; w < width; ++w) {
            let value = pixels[h * width * imageChannels + w * imageChannels + (channels - c - 1)];
            if (nchwFlag) {
              tensor[c * width * height + h * width + w] = (value - mean[c]) / std[c];
            } else {
              tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
            }
          }
        }
      }
    } else {
      throw new Error(`Unsupport '${channelScheme}' Color Channel Scheme `);
    }
  };

  /**
   * This method is to get downsample audio buffer.
   * @param {!Float32Array} buffer
   * @param {number} rate
   * @param {number} baseRate
   * @returns {!Float32Array}
   */
  _downsampleAudioBuffer = (buffer, rate, baseRate) => {
    if (rate == baseRate) {
      return buffer;
    }

    if (baseRate > rate) {
      throw "downsampling rate show be smaller than original sample rate";
    }

    const sampleRateRatio = Math.round(rate / baseRate);
    const newLength = Math.round(buffer.length / sampleRateRatio);
    let abuffer = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;

    while (offsetResult < abuffer.length) {
      let nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      abuffer[offsetResult] = accum / count;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return abuffer;
  };

  /**
   * This method is to get audio mfccs array.
   * @param {!Float32Array} pcm
   * @param {number} sampleRate
   * @param {number} windowSize
   * @param {number} windowStride
   * @param {number=} upperFrequencyLimit
   * @param {number=} lowerFrequencyLimit
   * @param {number=} filterbankChannelCount
   * @param {number=} dctCoefficientCount
   * @returns {!Array<number>}
   */
  _getAudioMfccs = (pcm, sampleRate, windowSize, windowStride,
                    upperFrequencyLimit = 4000,
                    lowerFrequencyLimit = 20,
                    filterbankChannelCount = 40,
                    dctCoefficientCount = 13) => {
    let pcmPtr = Module._malloc(8 * pcm.length);
    let lenPtr = Module._malloc(4);

    for (let i = 0; i < pcm.length; i++) {
      Module.HEAPF64[pcmPtr / 8 + i] = pcm[i];
    };

    Module.HEAP32[lenPtr / 4] = pcm.length;
    let tfMfccs = Module.cwrap('tf_mfccs', 'number',
          ['number', 'number', 'number', 'number',
           'number', 'number', 'number', 'number', 'number']);
    let mfccsPtr = tfMfccs(pcmPtr, lenPtr, sampleRate, windowSize,
          windowStride, upperFrequencyLimit, lowerFrequencyLimit,
          filterbankChannelCount, dctCoefficientCount);
    let mfccsLen = Module.HEAP32[lenPtr >> 2];
    let audioMfccs = [mfccsLen];

    for (let i = 0; i < mfccsLen; i++) {
      audioMfccs[i] = Module.HEAPF64[(mfccsPtr >> 3) + i];
    }

    Module._free(pcmPtr, lenPtr, mfccsPtr);
    return audioMfccs;
  };

  /**
   * This method is to set '_inputTensor' with audio input.
   * @param {!Object<string, *>}
   *     input = { // for Speech Command example
   *       src: {!HTMLElement}, // audio element
   *       options: {
   *         inputSize: {!Array<number>},
   *         sampleRate: {number},
   *         mfccsOptions: {!Object<string, *>}, // see details of mfccsOptions from speechCommandModels configurations of modelZoo.js
   *       },
   *     };
   */
  _getTensorByAudio = async (input) => {
    const audio = input.src;
    const options = input.options;
    const sampleRate = options.sampleRate;
    const mfccsOptions = options.mfccsOptions;
    const inputSize = options.inputSize.reduce((a, b) => a * b);
    let tensor = this._inputTensor[0];
    let audioContext = new (window.AudioContext || window.webkitAudioContext)();
    let rate = audioContext.sampleRate;

    let request = new Request(audio.src);
    let response = await fetch(request);
    let audioFileData = await response.arrayBuffer();
    let audioDecodeData = await audioContext.decodeAudioData(audioFileData);
    let audioPCMData = audioDecodeData.getChannelData(0);
    let abuffer = this._downsampleAudioBuffer(audioPCMData, rate, sampleRate);

    if (typeof mfccsOptions !== 'undefined') {
      abuffer = this._getAudioMfccs(abuffer,
                                    sampleRate,
                                    mfccsOptions.windowSize,
                                    mfccsOptions.windowStride,
                                    mfccsOptions.upperFrequencyLimit,
                                    mfccsOptions.lowerFrequencyLimit,
                                    mfccsOptions.filterbankChannelCount,
                                    mfccsOptions.dctCoefficientCount);
    }

    if (abuffer.length >= inputSize) {
      for (let i = 0; i < inputSize; i++) {
        tensor[i] = abuffer[i];
      }
    } else {
      for (let i = 0; i < abuffer.length; i++) {
        tensor[i] = abuffer[i];
      }
    }
  };

  /** @override */
  _getInputTensor = async (input) => {
    if (input.src.tagName === 'AUDIO') {
      await this._getTensorByAudio(input);
    } else {
      this._getTensor(input);
    }
  };

  /** @override */
  _doInference = async () => {
    let status = await this._model.compute(this._inputTensor, this._outputTensor);
    console.log(`Computed Status: [${status}]`);
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

  /** @override */
  _getOutputTensor = () => {
    return this._outputTensor[0];
  };

  /**
   * This method is to free allocated memory resources for model compilation process by polyfill backend.
   */
  _freeAllocatedMemory = () => {
    if (this._currentBackend != 'WebML') {
      // free allocated memory on compilation process by polyfill WASM / WebGL backend.
      if (this._model && this._model._compilation && this._model._compilation._preparedModel) {
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
   * @param {object} configs An array object for config array, likes:
   *     [{backend: 'WASM', prefer: 'fast'}]
   *     [{backend: 'WebGL', prefer: 'sustained'}]
   *     [{backend: 'WebGPU', prefer: 'sustained'}]
   *     [{backend: 'WebML', prefer: 'fast'}]
   *     [{backend: 'WebML', prefer: 'sustained'}]
   *     or
   *     [{backend: 'WASM', prefer: 'fast'}, {backend: 'WebGL', prefer: 'sustained'}]
   *     [{backend: 'WASM', prefer: 'fast'}, {backend: 'WebML', prefer: 'fast'}]
   *     etc.
   * @param {!Array<number>|undefined} layerList An array object that for layer array, likes: [1, 2 ,3], or undefined that means an array with all layers indexes.
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
        isQuantized: this._currentModelInfo.isQuantized || false,
        isIE: this._currentModelInfo.isIE || false,
        isDNN: this._currentModelInfo.isDNN || false,
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
