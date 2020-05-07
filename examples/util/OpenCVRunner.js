class OpenCVRunner extends BaseRunner {
  constructor() {
    super();
    this._output = null;
  }

  /**
   * This method is to load model file with specified url.
   * @param url: A string for model file url.
   * @returns {string} This returns status of load model file, ['ERROR' | 'SUCCESS'].
   */
  _loadModelFile = async (url) => {
    let status = 'ERROR';

    if (url !== undefined) {
      const arrayBuffer = await this._loadURL(url, this._progressHandler, true);
      const bytes = new Uint8Array(arrayBuffer);
      switch (url.split('.').pop()) {
        case 'onnx':
          const err = onnx.ModelProto.verify(bytes);
          if (err) {
            throw new Error(`The model file ${url} is invalid, ${err}`);
          }
          try {
            cv.FS_createDataFile('/', this._currentModelInfo.modelId, bytes, true, false, false);
          } catch (e) {
            if (e.errno === 17) {
              console.log(`${this._currentModelInfo.modelId} already exited.`);
              status = 'SUCCESS'
            } else {
              console.error(e);
            }
          }
          break;
        default:
          throw new Error(`Unrecognized model format, support TFLite | ONNX | OpenVINO model`);
      }
    } else {
      throw new Error(`There's none model file info, please check config info of modelZoo.`);
    }

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
    this._setModelInfo(modelInfo);
    await this._loadModelFile(this._currentModelInfo.modelFile);

    if (this._currentModelInfo.labelsFile != null) {
      await this._loadLabelsFile(this._currentModelInfo.labelsFile);
    }
  };

  /**
   * This method is to compile a machine learning model for OpenCV.js.
   * @param options: A string, not used.
   * @returns {string}. This returns status of compilation model.
   */
  compileModel = (options) => {
    if (!this._bLoaded) {
      return 'NOT_LOADED';
    }

    if (this._bInitialized) {
      return 'INITIALIZED';
    }

    this._setInitializedFlag(false);
    let model = null;

    switch (this._currentModelInfo.format) {
      case 'ONNX':
        model = cv.readNetFromONNX(this._currentModelInfo.modelId);
        break;
      default:
        throw new Error(`Unsupported '${this._currentModelInfo.format}' input`);
    }

    this._setModel(model);
    this._setInitializedFlag(true);
    return 'SUCCESS';
  };

  /**
   * This method is to get input tensor with input src (HTML [<img> | <video>] element).
   * @param src: An object for HTML [<img> | <video>] element.
   * @param options: A string to get inputTensor. Details:
   * options = {
   *   // inputSize was configed in modelZoo.js, inputSize = [h, w, c].
   *   inputSize: inputSize,
   *   // preOptions was also configed in modelZoo.js,
   *   // preOptions= {mean:[number,number,number,number],std:[number,number,number,number]}
   *   preOptions: preOptions,
   *   imageChannels: 4,
   * };
   * @returns {object} This returns an object for input tensor.
   */
  _getInputTensor = (src, options) => {
    const mean = options.preOptions.mean;
    const std = options.preOptions.std;
    const [sizeW, sizeH, channels] = options.inputSize;
    const imageChannels = options.imageChannels;
    const width = src.videoWidth || src.naturalWidth;
    const height = src.videoHeight || src.naturalHeight;
    const canvasElement = document.createElement('canvas');
    canvasElement.width = width;
    canvasElement.height = height;
    const canvasContext = canvasElement.getContext('2d');
    canvasContext.drawImage(src, 0, 0, width, height);
    const pixels = canvasContext.getImageData(0, 0, width, height).data;
    let stddata = [];
    for (let c = 0; c < channels; ++c) {
      for (let h = 0; h < height; ++h) {
        for (let w = 0; w < width; ++w) {
          let value = pixels[h * width * imageChannels + w * imageChannels + c];
          stddata[h * width * channels + w * channels + c] = (value / 255 - mean[c]) / std[c];
        }
      }
    }
    let inputMat = cv.matFromArray(height, width, cv.CV_32FC3, stddata);
    let input = cv.blobFromImage(inputMat, 1, new cv.Size(sizeW, sizeH), new cv.Scalar(0, 0, 0));
    return input;
  };

  /**
   * This method is to do inference with input src (HTML [<img> | <video>] element).
   * @param src: An object for HTML [<img> | <video>] element.
   * @param options: A string to get inputTensor, will ba a parameter to the method '_getInputTensor'. Details:
   * options = {
   *   // inputSize was configed in modelZoo.js, inputSize = [h, w, c].
   *   inputSize: inputSize,
   *   // preOptions was also configed in modelZoo.js,
   *   // preOptions= {mean:[number,number,number,number],std:[number,number,number,number]}
   *   preOptions: preOptions,
   *   imageChannels: 4,
   * };
   * @returns {string} This returns a string for inference status.
   */
  run = (src, options) => {
    let status = 'ERROR';
    let input = this._getInputTensor(src, options);
    this._model.setInput(input);
    const start = performance.now();
    this._output = this._model.forward();
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    status = 'SUCCESS';
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${delta} ms]`);
    return status;
  };
}
