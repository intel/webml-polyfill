class ImageClassificationOpenCVRunner extends OpenCVRunner {
  constructor() {
    super();
  }

  _getOutputTensor = (output) => {
    output.outputTensor = softmax(this._output.data32F);
    this._output.delete();
  };
}
