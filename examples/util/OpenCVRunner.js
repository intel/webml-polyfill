class OpenCVRunner extends BaseRunner {
  constructor() {
    super();
    this._output = null;
  }

  _getRawModel = async (url) => {
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

  _getOtherResources = async () => {
    // Override by inherited if needed, likes load labels file
  };

  _getModelResources = async () => {
    await this._getRawModel(this._currentModelInfo.modelFile);
    await this._getOtherResources();
  };

  loadModel = async (modelInfo) => {
    if (this._bLoaded && this._currentModelInfo.modelFile === modelInfo.modelFile) {
      return 'LOADED';
    }

    // reset all states
    this._setLoadedFlag(false);
    this._setInitializedFlag(false);
    this._setModelInfo(modelInfo);
    await this._getModelResources();
  };

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

  _getInputTensor = (src, options) => {
    const mean = options.preOptions.mean;
    const std = options.preOptions.std;
    const [width, height, channel] = options.inputSize;
    let mat = cv.imread(src.id);
    let matC3 = new cv.Mat(mat.matSize[0], mat.matSize[1], cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2RGB);
    let matdata = matC3.data;
    let stddata = [];
    for(let i = 0; i < mat.matSize[0] * mat.matSize[1]; ++i) {
      for (let c = 0; c < channel; ++c) {
        stddata.push((matdata[channel * i + c] / 255 - mean[c]) / std[c]);
      }
    };
    let inputMat = cv.matFromArray(mat.matSize[0], mat.matSize[1], cv.CV_32FC3, stddata);
    let input = cv.blobFromImage(inputMat, 1, new cv.Size(width, height), new cv.Scalar(0, 0, 0));
    return input;
  };

  run = (src, options) => {
    let status = 'ERROR';
    let input = this._getInputTensor(src, options);
    this._model.setInput(input);
    const start = performance.now();
    this._output = this._model.forward();
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${delta} ms]`);
    return status;
  };

  getOutput = () => {
    let output = {
      inferenceTime: this._inferenceTime,
    };
    this._updateOutput(output); // add custom output info
    return output;
  };
}
