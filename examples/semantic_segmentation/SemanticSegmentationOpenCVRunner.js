class SemanticSegmentationOpenCVRunner extends OpenCVRunner {
  constructor() {
    super();
  }

  /** @override */
  _getOutputTensorTypedArray = () => {
    return Int32Array;
  };

  /** @override */
  _getOutputTensor = () => {
    let outputTensor = this._outputArgmax(this._output);
    this._output.delete();
    return outputTensor;
  };

  /**
   * This method is to process the output using argmax function.
   * @param {Mat} output The output of the dnn_Net.
   */
  _outputArgmax = (output) => {
    const C = output.matSize[1];
    const H = output.matSize[2];
    const W = output.matSize[3];
    const outputData = output.data32F;
    const imgSize = H * W;

    let outputTensor = [];
    for (let i = 0; i < imgSize; ++i){
      let id = 0;
      for (let j = 0; j < C; ++j) {
        if (outputData[j * imgSize + i] > outputData[id * imgSize + i]) {
          id = j;
        }
      }
      outputTensor.push(id);
    }

    return outputTensor;
  }

}
