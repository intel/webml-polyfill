class OpenCVExecutor extends BaseExecutor {
  constructor() {
    super();
    this._currentBackend;
  }

  setBackend = (backend) => {
    this._currentBackend = backend;
  };

  /** @override */
  _createRunner = () => {
    const runner = new OpenCVRunner();
    return runner;
  };

  /** @override */
  _compileModel = async () => {
    await this._runner.compileModel();
  };

  /** @override */
  execute = async (iterations, logger) => {
    this._inferenceTimeList = [];
    try {
      let inferenceOutput;
      for (let i = 0;  i < iterations; i++) {
        // logger.log(`Iteration: ${i + 1} / ${iterations}`);
        // await new Promise(resolve => requestAnimationFrame(resolve));
        await this._executeSingle();
        inferenceOutput = this._runner.getOutput();
        this._inferenceTimeList.push(inferenceOutput.inferenceTime);
      }
      this._postProcess(inferenceOutput);
    } catch (e) {
      console.error(e);
    };
  };
}
