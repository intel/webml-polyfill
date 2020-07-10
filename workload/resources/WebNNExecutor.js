class WebNNExecutor extends BaseExecutor {
  constructor() {
    super();
    this._currentBackend;
    this._currentPrefer;
    this._bEagerMode = false;
    this._supportedOps = [];
  }

  setBackend = (backend) => {
    this._currentBackend = backend;
  };

  setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  /**
   * This method is to set '_bEagerMode'.
   * @param {boolean} flag
   */
  setEagerMode = (flag) => {
    this._bEagerMode =  flag;
  };

  /**
   * This method is to set '_supportedOps'.
   * @param {!Array<number>} ops
   */
  setSupportedOps = (ops) => {
    this._supportedOps = ops;
  };

  /** @override */
  _createRunner = () => {
    const runner = new WebNNRunner();
    return runner;
  };

  /** @override */
  _compileModel = async () => {
    await this._runner.compileModel({
      backend: this._currentBackend.replace('WebNN', 'WebML'),
      prefer: this._currentPrefer,
      eagerMode: this._bEagerMode,
      supportedOps: this._supportedOps
    });
  };

  /**
   * This method is to get profilling results by each op for WASM / WebNN backend.
   * @returns {!Array} This returns an array oject of profilling results.
   */
  _getProfilingResults = () => {
    let profilingResults = null;
    if (this._currentBackend !== 'WebNN') {
      profilingResults = [this._runner._model._compilation._preparedModel.dumpProfilingResults()];
    }
    return profilingResults;
  };

  /** @override */
  execute = async (iterations, logger) => {
    this._inferenceTimeList = [];
    try {
      for (let i = 0;  i < iterations; i++) {
        // logger.log(`Iteration: ${i + 1} / ${iterations}`);
        // await new Promise(resolve => requestAnimationFrame(resolve));
        await this._executeSingle();
        let inferenceOutput = this._runner.getOutput();
        this._inferenceTimeList.push(inferenceOutput.inferenceTime);
      }
      this._postProcess();
    } catch (e) {
      console.error(e);
    };
  };

  /** @override */
  getInferenceResults = () => {
    return {
      inferenceTimeList: this._inferenceTimeList,
      decodeTime: this._decodeTime,
      profiling: this._getProfilingResults(),
    };
  };
}
