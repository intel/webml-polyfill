class BaseRunner {
  constructor() {
    this._currentBackend = null;
    this._currentPrefer = null;
    this._currentModelInfo = {};
    this._inputTensor = [];
    this._outputTensor = [];
    this._currentRequest = null;
    this._progressHandler = null;
    this._rawModel = null;
    this._bLoaded = false; // loaded status of raw model for Web NN API
    this._model = null; // get Web NN model by converting raw model
    this._subgraphsSummary = [];
    this._modelRequiredOps = null;
    this._deQuantizeParams = null;
    this._bInitialized = false; // initialized status for model
    this._bEagerMode = false;
    this._supportedOps = new Set();
    this._inferenceTime = 0.0; // ms
    this._labels = null;
  }

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo
  };

  _setRequest = (req) => {
    // Record current request, aborts the request if it has already been sent
    this._currentRequest = req;
  };

  setProgressHandler = (handler) => {
    // Use handler to prompt for model loading progress info
    this._progressHandler = handler;
  };

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  _setRawModel = (rawModel) => {
    this._rawModel = rawModel;
  };

  _setLoadedFlag = (flag) => {
    this._bLoaded = flag;
  };

  _setModel = (model) => {
    this._model = model;
  };

  _setSubgraphsSummary = (summary) => {
    this._subgraphsSummary = summary;
  };

  _setModelRequiredOps = (ops) => {
    this._modelRequiredOps = ops;
  };

  _setDeQuantizeParams = (params) => {
    this._deQuantizeParams = params;
  };

  _setInitializedFlag = (flag) => {
    this._bInitialized = flag;
  };

  _setInferenceTime = (t) => {
    this._inferenceTime = t;
  };

  _loadURL = async (url, handler = null, isBinary = false) => {
    let _this = this;
    return new Promise((resolve, reject) => {
      if (_this._currentRequest != null) {
        _this._currentRequest.abort();
      }
      let oReq = new XMLHttpRequest();
      _this._setRequest(oReq);
      oReq.open('GET', url, true);
      if (isBinary) {
        oReq.responseType = 'arraybuffer';
      }
      oReq.onload = function (ev) {
        _this._setRequest(null);
        if (oReq.readyState === 4) {
          if (oReq.status === 200) {
            resolve(oReq.response);
          } else {
            reject(new Error(`Failed to load ${url} . Status: [${oReq.status}]`));
          }
        }
      };
      if (handler != null) {
        oReq.onprogress = handler;
      }
      oReq.send();
    });
  };

  _setLabels = (labels) => {
    this._labels = labels;
  };

  _loadLabelsFile = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

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
        default:
          throw new Error(`Unrecognized model format, support TFLite | ONNX | OpenVINO model`);
      }
    } else {
      throw new Error(`There's none model file info, please check config info of modelZoo.`);
    }

    this._setRawModel(rawModel);
    this._setLoadedFlag(true);
  };

  loadModel = async (modelInfo) => {
    if (this._bLoaded && this._currentModelInfo.modelFile === modelInfo.modelFile) {
      console.log(`${this._currentModelInfo.modelFile} already loaded.`);
      return;
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
    await this._loadLabelsFile(this._currentModelInfo.labelsFile);
  };

  _getInputTensorTypedArray = () => {
    // Override by inherited if needed
    if(this._currentModelInfo) {
      const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
      return typedArray;
    }
  };

  _initInputTensor = () => {
    if(this._currentModelInfo) {
      const typedArray = this._getInputTensorTypedArray();
      this._inputTensor = [new typedArray(this._currentModelInfo.inputSize.reduce((a, b) => a * b))];
    }
  };

  _getOutputTensorTypedArray = () => {
    // Override by inherited if needed
    if(this._currentModelInfo) {
      const typedArray = this._currentModelInfo.isQuantized || false ? Uint8Array : Float32Array;
      return typedArray;
    }
  };

  _initOutputTensor = () => {
    // Override by inherited if needed
    if(this._currentModelInfo) {
      const typedArray = this._getOutputTensorTypedArray();
      const outputSize = this._currentModelInfo.outputSize;

      if (typeof outputSize === 'number') {
        this._outputTensor = [new typedArray(outputSize)];
      } else {
        this._outputTensor = [new typedArray(outputSize.reduce((a, b) => a * b))];
      }
    }
  };

  compileModel = async (backend, prefer) => {
    if (this._bInitialized && backend === this._currentBackend && prefer === this._currentPrefer) {
      console.log('Model was already compiled.');
      return;
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
        throw new Error(`Unsupported '${rawModel._rawFormat}' input.`);
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
  };

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

    let canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    let canvasContext = canvasElement.getContext('2d');

    if (drawOptions) {
      canvasContext.drawImage(image, drawOptions.sx, drawOptions.sy, drawOptions.sWidth, drawOptions.sHeight,
        0, 0, drawOptions.dWidth, drawOptions.dHeight);
    } else {
      if (options.scaledFlag) {
        const resizeRatio = Math.max(Math.max(image.width, image.height) / width, 1);
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
              tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else if (channels === 1) {
        for (let c = 0; c < channels; ++c) {
          for (let h = 0; h < height; ++h) {
            for (let w = 0; w < width; ++w) {
              let index = h * width * imageChannels + w * imageChannels + c;
              let value = (pixels[index] + pixels[index + 1] + pixels[index + 2]) / 3;
              tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
            }
          }
        }
      }
    } else if (channelScheme === 'BGR') {
      for (let c = 0; c < channels; ++c) {
        for (let h = 0; h < height; ++h) {
          for (let w = 0; w < width; ++w) {
            let value = pixels[h * width * imageChannels + w * imageChannels + (channels - c - 1)];
            tensor[h * width * channels + w * channels + c] = (value - mean[c]) / std[c];
          }
        }
      }
    } else {
      throw new Error(`Unsupport '${channelScheme}' Color Channel Scheme `);
    }
  };

  run = async (input) => {
    this._getTensor(input);

    const start = performance.now();
    let status = await this._model.compute(this._inputTensor, this._outputTensor);
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${delta} ms]`);
  };

  getRequiredOps = () => {
    return this._modelRequiredOps;
  };

  getSubgraphsSummary = () => {
    return this._subgraphsSummary;
  };

  getDeQuantizeParams = () => {
    return this._deQuantizeParams;
  };

  getOutput = () => {
    let output = {
      tensor: this._outputTensor[0],
      inferenceTime: this._inferenceTime,
      labels: this._labels,
    };

    return output;
  };

  deleteAll = () => {
    if (this._currentBackend != 'WebML') {
      // free allocated memory on compilation process by polyfill WASM / WebGL backend.
      if (this._model._compilation && this._model._compilation._preparedModel) {
        this._model._compilation._preparedModel._deleteAll();
      }
    }
  };

  // for debugging
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

class SemanticSegmentationRunner extends BaseRunner {
  constructor() {
    super();
  }

  _getOutputTensorTypedArray = () => {
    return Int32Array;
  };
}

export { BaseRunner, SemanticSegmentationRunner }

