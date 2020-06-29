class SkeletonDetectionRunner {
  constructor() {
    this._currentModelInfo = {};
    this._modelConfig;
    this._currentBackend;
    this._currentPrefer;
    this._model;
    this._cacheMap = new Map();
    this._bInitialized = false; // initialized status for model
    this._inputTensor = null;
    this._heatmapTensor = null;
    this._offsetTensor = null;
    this._displacementFwd = null;
    this._displacementBwd = null;
    this._supportedOps =  new Set();
    this._bEagerMode = false;
    this._inferenceTime = 0.0;  // ms
  }

  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo;
  };

  _setModelConfig = (options) => {
    this._modelConfig = JSON.parse(JSON.stringify(options));
  };

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  _setInitializedFlag = (flag) => {
    this._bInitialized = flag;
  };

  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  setEagerMode = (flag) => {
    this._bEagerMode = flag;
  };

  _setInferenceTime = (t) => {
    this._inferenceTime = t;
  } ;

  loadAndCompileModel = async (backend, prefer, modelInfo, modelConfig, workload) => {
    if (this._bInitialized
        && backend === this._currentBackend
        && prefer === this._currentPrefer
        && modelConfig.version === this._modelConfig.version
        && modelConfig.outputStride === this._modelConfig.outputStride
        && modelConfig.scaleFactor === this._modelConfig.scaleFactor
        && modelConfig.useAtrousConv === this._modelConfig.useAtrousConv) {
      console.log('Model already loaded and was compiled.');
      return;
    }

    this._freeAllocatedMemory();
    this._setInitializedFlag(false);
    this._setBackend(backend);
    this._setPrefer(prefer);
    this._setModelInfo(modelInfo);
    this._setModelConfig(modelConfig);
    let version = Number(this._modelConfig.version);
    const useAtrousConv = this._modelConfig.useAtrousConv;
    const outputStride = Number(this._modelConfig.outputStride);
    const scaleFactor = this._modelConfig.scaleFactor;
    const modelArch = ModelArch.get(version);
    const inputSize = this._currentModelInfo.inputSize;
    const scaleWidth = getValidResolution(scaleFactor, inputSize[1], outputStride);
    const scaleHeight = getValidResolution(scaleFactor, inputSize[0], outputStride);
    const scaleSize = [1, scaleWidth, scaleHeight, 3];
    if (typeof workload !== 'undefined') {
      version = {'version': version, 'adjustPath': true,};
    }
    this._model = new PoseNet(modelArch, version, useAtrousConv, outputStride,
                              scaleSize, this._cacheMap, this._currentBackend, this._currentPrefer);
    this._model.setSupportedOps(this._supportedOps);
    this._model.setEagerMode(this._bEagerMode);
    const result = await this._model.createCompiledModel();
    console.log(`Created and compiled model status: [${result}]`);

    this._inputTensor = [new Float32Array(scaleSize.reduce((a, b) => a * b))];
    let heatmapTensorSize;

    if ((this._modelConfig.version == 0.75 || this._modelConfig.version == 0.5) && this._modelConfig.outputStride == 32) {
      heatmapTensorSize = product(toHeatmapsize(scaleSize, 16));
    } else {
      heatmapTensorSize = product(toHeatmapsize(scaleSize, this._modelConfig.outputStride));
    }

    this._heatmapTensor = new Float32Array(heatmapTensorSize);
    this._offsetTensor = new Float32Array(heatmapTensorSize * 2);
    this._displacementFwd = new Float32Array(heatmapTensorSize / 17 * 32);
    this._displacementBwd = new Float32Array(heatmapTensorSize / 17 * 32);
    const start = performance.now();
    await this._model.compute(this._inputTensor[0], this._heatmapTensor,
                              this._offsetTensor, this._displacementFwd,
                              this._displacementBwd);
    const delta = performance.now() - start;
    console.log(`Warm up Time: [${delta} ms]`);
    this._setInitializedFlag(true);
  }

  getRequiredOps = () => {
    return this._model.getRequiredOps();
  };

  getSubgraphsSummary = () => {
    if (this._currentBackend !== 'WebML'
        && this._model._compilation
        && this._model._compilation._preparedModel) {
      return this._model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  };

  _getInputTensor = (input) => {
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

  _doInference = async () => {
    await this._model.compute(this._inputTensor[0], this._heatmapTensor,
                              this._offsetTensor, this._displacementFwd,
                              this._displacementBwd);
  };

  run = async (input) => {
    if (!this._bInitialized) return;

    this._getInputTensor(input);
    const start = performance.now();
    await this._doInference();
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Compute Time: [${delta} ms]`);
  };

  getOutput = () => {
    const output = {
      heatmapTensor: this._heatmapTensor,
      offsetTensor: this._offsetTensor,
      displacementFwd: this._displacementFwd,
      displacementBwd: this._displacementBwd,
      inferenceTime: this._inferenceTime,
    };

    return output;
  };

  /**
   * This method is to free allocated memory resources for model compilation process by polyfill backend.
   */
  _freeAllocatedMemory = () => {
    if (this._currentBackend != 'WebML') {
      // free allocated memory on compilation process by polyfill WASM / WebGL backend.
      if (this._model
          && this._model._compilation
          && this._model._compilation._preparedModel) {
        this._model._compilation._preparedModel._deleteAll();
      }
    }
  };
}
