class BaseRunner {
  constructor() {
    this._currentModelInfo = {};
    this._inputTensor = [];
    this._currentRequest = null;
    this._progressHandler = null;
    this._bLoaded = false; // loaded status of raw model for Web NN API
    this._model = null; // get Web NN model by converting raw model
    this._bInitialized = false; // initialized status for model
    this._inferenceTime = 0.0; // ms
  }

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

  _setLoadedFlag = (flag) => {
    this._bLoaded = flag;
  };

  _setModel = (model) => {
    this._model = model;
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

  _getOtherResources = async () => {
    // Override by inherited if needed, likes load labels file
  };

  _getModelResources = async () => {
    await this._getRawModel(this._currentModelInfo.modelFile);
    await this._getOtherResources();
  };

  loadModel = async (modelInfo) => {
    // Override by inherited if needed
  };

  compileModel = async (options) => {
    // Override by inherited if needed
  };

  run = async (src, options) => {
    // Override by inherited if needed
  };

  _updateOutput = (output) => {
    // Override by inherited if needed
  };

  getOutput = () => {
    let output = {
      inferenceTime: this._inferenceTime,
    };
    this._updateOutput(output); // add custom output info
    return output;
  };
}
