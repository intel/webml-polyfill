class FaceDetectorRunner extends WebNNRunner {
  constructor() {
    super();
    this._outputBoxTensor;
    this._outputClassScoresTensor;
  }

  _initOutputTensor = () => {
    if (this._currentModelInfo.category === 'SSD') {
      // SSD models
      const totalBoxes = this._currentModelInfo.numBoxes.reduce((a, b) => a + b);
      const boxTensorLen = totalBoxes * this._currentModelInfo.boxSize;
      const classTensorLen = totalBoxes * this._currentModelInfo.numClasses;
      this._outputBoxTensor = new Float32Array(boxTensorLen);
      this._outputClassScoresTensor = new Float32Array(classTensorLen);
      const options = {
        numBoxes: this._currentModelInfo.numBoxes,
        boxSize: this._currentModelInfo.boxSize,
        numClasses: this._currentModelInfo.numClasses,
      };
      this._outputTensor = prepareOutputTensorSSD(this._outputBoxTensor, this._outputClassScoresTensor, options);
    } else {
      // YOLO models
      this._outputTensor = [new Float32Array(this._currentModelInfo.outputSize)];
    }
  };

  _getOutputTensor = () => {
    let outputTensor = {};

    if (this._currentModelInfo.category === 'SSD') {
      outputTensor.outputBoxTensor = this._outputBoxTensor;
      outputTensor.outputClassScoresTensor = this._outputClassScoresTensor;
    } else {
      // YOLO models
      outputTensor = this._outputTensor[0];
    }

    return outputTensor;
  };
}