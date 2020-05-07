class ImageClassificationOpenCVRunner extends OpenCVRunner {
  constructor() {
    super();
  }

  _passOutputTensor = (output) => {
    output.outputTensor = softmax(this._output.data32F);
  };
}