class BaseApp {
  constructor(models) {
    this._inferenceModels = models;
    this._currentModelId = 'none';
    // currentInputElement: an element of HTMLImageElement | HTMLVideoElement | HTMLAudioElement
    this._currentInputElement = null;
    // Backend type: 'WASM' | 'WebGL' | 'WebML'
    this._currentBackend = 'WASM';
    // Prefer type: 'none' | 'fast' | 'sustained' | 'low'
    this._currentPrefer = 'none';
    // Runner instance to load raw model, convert raw model to Web NN model, then model does compilation, execution
    // One App could have multi different runner instances
    this._runner = null;
  }

  _setModelId = (modelIdStr) => {
    this._currentModelId = modelIdStr;
  };

  _setInputElement = (element) => {
    this._currentInputElement = element;
  };

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  _getRunner = () => {
    // Override by inherited
  };

  _loadModel = async () => {
    // Override by inherited
  };

  _compileModel = async () => {
    // Override by inherited
  };

  _predict = async () => {
    // Override by inherited
  };

  _postProcess = () => {
    // Override by inherited
  };

  UI = () => {
    // Override by inherited
    // 1._setInputElement
    // 2.ready for runner: _setModelId -> _setBackend -> _setPrefer
  };

  main = async () => {
    // Override by inherited
    // _getRunner -> _loadModel -> _compileModel -> _predict -> _postProcess
  };
};
