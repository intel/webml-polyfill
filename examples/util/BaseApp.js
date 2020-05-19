class BaseApp {
  constructor(models) {
    this._inferenceModels = models;
    this._currentModelId = 'none';
    // currentInputElement: an element of HTMLImageElement | HTMLVideoElement | HTMLAudioElement
    this._currentInputElement = null;
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

  /**
   * This method is to get runner instance by calling '_createRunner' method.
   */
  _getRunner = () => {};

  /**
   * This method is for loading model file [and label file if label information is required].
   */
  _loadModel = async () => {};

  /**
   * This method is to compile a machine learning model.
   */
  _compileModel = async () => {};

  /**
   * This method is to run inference by model.
   */
  _predict = async () => {};

  /**
   * This method is to do post processing with inference result.
   */
  _postProcess = () => {};

  /**
   * This method is to run inference and do post processing.
   */
  main = async () => {
    // Override by inherited
    // _getRunner -> _loadModel -> _compileModel -> _predict -> _postProcess
  };
};
