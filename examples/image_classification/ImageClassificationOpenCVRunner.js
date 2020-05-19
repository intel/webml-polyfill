class ImageClassificationOpenCVRunner extends OpenCVRunner {
  constructor() {
    super();
  }

  /** @override */
  _getOutputTensor = () => {
    let outputTensor = softmax(this._output.data32F);
    this._output.delete();
    return outputTensor;
  };
}
