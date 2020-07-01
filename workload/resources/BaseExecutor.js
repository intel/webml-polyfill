class BaseExecutor {
  constructor() {
    this._currentInputElement;
    this._currentCategory;
    this._currentModelInfo;
    this._runner;
    this._inferenceTimeList = [];
    this._decodeTime = 0.0;
  }

  setInputElement = (element) => {
    this._currentInputElement = element;
  };

  setCategory = (category) => {
    this._currentCategory = category;
  };

  setModelId = (modelId) => {
    this._currentModelId = modelId;
  };

  /**
   * This method is to set '_currentModelInfo'.
   * @param {!Object<string, *>} modelInfo An object that for model info which was configed in modeZoo.js.
   *     An example for model info:
   *       modelInfo = {
   *         modelName: {string}, // 'MobileNet v1 (TFLite)'
   *         format: {string}, // 'TFLite'
   *         modelId: {string}, // 'mobilenet_v1_tflite'
   *         modelSize: {string}, // '16.9MB'
   *         inputSize: {!Array<number>}, // [224, 224, 3]
   *         outputSize: {number} // 1001
   *         modelFile: {string}, // '../image_classification/model/mobilenet_v1_1.0_224.tflite'
   *         labelsFile: {string}, // '../image_classification/model/labels1001.txt'
   *         preOptions: {!Obejct<string, *>}, // {mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5],}
   *         intro: {string}, // 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
   *         paperUrl: {string}, // 'https://arxiv.org/pdf/1704.04861.pdf'
   *       };
   */
  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo;
  };

  _setDecodeTime = (t) => {
    this._decodeTime = t;
  };

  /**
   * This method returns runner instance to load model/compile model/inference.
   * @returns {object} This returns a runner instance.
   */
  _createRunner = () => {}

  /**
   * This method is to get runner instance by calling '_createRunner' method.
   */
  getRunner = () => {
    // Override by inherited when example has co-work runners
    if (this._runner == null) {
      this._runner = this._createRunner();
    }
  };

  /**
   * This method is for loading model file [and label file if label information is required].
   * @param {String} modelId: A String that for inference model Id.
   * @param {String} coModelId: A String that for inference co-model Id.
   */
  _loadModel = async (modelId, coModelId) => {
    // Override by inherited when example has co-work runners
    const modelInfo = getModelById(SUPPORTED_WORKLOAD_CATG[this._currentCategory].model, modelId);

    if (modelInfo != null) {
      this._setModelInfo(modelInfo);
      await this._runner.loadModel(modelInfo, 'workload');
    } else {
      throw new Error('Unrecorgnized model, please check your typed url.');
    }
  };

  /**
   * This method is to compile a machine learning model.
   */
  _compileModel = async () => {};

  /**
   * This method is for loading model file [and label file if label information
   * is required] and compilation a machine learning model.
   * @param {String} modelId: A String that for inference model Id.
   * @param {String} coModelId: A String that for inference co-model Id.
   */
  loadAndCompileModel = async (modelId, coModelId) => {
    await this._loadModel(modelId, coModelId);
    await this._compileModel();
  };

  /**
   * This method is to do post processing with inference result.
   * @param {Object} data: An object that for output results.
   */
  _postProcess = (data) => {};

  /**
   * This method is to run inference by model.
   */
  _executeSingle = async () => {
    const input = {
      src: this._currentInputElement,
      options: {
        inputSize: this._currentModelInfo.inputSize,
        preOptions: this._currentModelInfo.preOptions,
        imageChannels: 4,
      },
    };
    await this._runner.run(input);
  };

  /**
   * This method is to run <iterations> times inference by model.
   * @param {number} iterations: A number that for iterations of inference.
   */
  execute = async (iterations) => {};

  /**
   * This method is to get inference results.
   */
  getInferenceResults = () => {
    return {
      inferenceTimeList: this._inferenceTimeList,
      decodeTime: this._decodeTime,
    };
  };
}
