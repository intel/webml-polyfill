class ImageClassificationOpenVINORunner extends OpenVINORunner {
  constructor() {
    super();
  }

  /** @override */
  _getOutputTensor = () => {
    const postSoftmax = this._postOptions.softmax || false;
    let outputTensor;
    if(postSoftmax) {
      outputTensor = softmax(this._output);
    } else {
      outputTensor = this._output;
    }
    return outputTensor;
  };
}