class ObjectDetectionRunner extends WebNNRunner {
  constructor() {
    super();
    this._labels = null;
    this._outputBoxTensor = [];
    this._outputClassScoresTensor = [];
    this._deQuantizedOutputBoxTensor = [];
    this._deQuantizedOutputClassScoresTensor = [];
  }

  _initOutputTensor = () => {
    if (this._currentModelInfo.category === 'SSD') {
      // SSD models
      const totalBoxes = this._currentModelInfo.numBoxes.reduce((a, b) => a + b);
      const boxTensorLen = totalBoxes * this._currentModelInfo.boxSize;
      const classTensorLen = totalBoxes * this._currentModelInfo.numClasses;
      let typedArray;
      if (this._currentModelInfo.isQuantized) {
        typedArray = Uint8Array;
        this._deQuantizedOutputBoxTensor = new Float32Array(boxTensorLen);
        this._deQuantizedOutputClassScoresTensor = new Float32Array(classTensorLen);
      } else {
        typedArray = Float32Array;
      }
      this._outputBoxTensor = new typedArray(boxTensorLen);
      this._outputClassScoresTensor = new typedArray(classTensorLen);
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

  _setLabels = (labels) => {
    this._labels = labels;
  };

  _getLabels = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  _getOtherResources = async () => {
    await this._getLabels(this._currentModelInfo.labelsFile);
  };

  _updateSubOutput = (output) => {
    output.labels = this._labels;
    output.outputBoxTensor = this._outputBoxTensor;
    output.outputClassScoresTensor = this._outputClassScoresTensor;

    if (this._currentModelInfo.isQuantized) {
      output.deQuantizedOutputBoxTensor = this._deQuantizedOutputBoxTensor;
      output.deQuantizedOutputClassScoresTensor = this._deQuantizedOutputClassScoresTensor;
    }
  };
}