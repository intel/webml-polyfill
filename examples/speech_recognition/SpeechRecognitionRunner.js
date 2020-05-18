class SpeechRecognitionRunner extends WebNNRunner {
  constructor() {
    super();
  }

  _getInputTensor = (input) => {
    this._inputTensor[0].set(input);
  };
}
