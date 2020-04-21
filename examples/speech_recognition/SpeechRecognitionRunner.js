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
}
