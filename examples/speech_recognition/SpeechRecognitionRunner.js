class SpeechRecognitionRunner extends BaseRunner {
  constructor() {
    super();
  }

  run = async (inputTensor) => {
    let status = 'ERROR';

    this._inputTensor[0].set(inputTensor);
    const start = performance.now();
    status = await this._model.compute(this._inputTensor, this._outputTensor);
    const delta = performance.now() - start;
    this._setInferenceTime(delta);

    return status;
  };

  compileModel = async (backend, prefer, config) => {
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
      inputScaleFactor: config.scaleFactor
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
    const compileStatus = await this._model.createCompiledModel();
    console.log(`Compilation Status: [${compileStatus}]`);

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
}
