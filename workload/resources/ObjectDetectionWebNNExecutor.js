class ObjectDetectionWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
  }

  /** @override */
  _createRunner = () => {
    const runner = new ObjectDetectionRunner();
    return runner;
  };

  /** @override */
  _postProcess = (data) => {
    let output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    let canvasShowElement = document.getElementById('showcanvas');
    const modelInfo = this._currentModelInfo;
    const options = {
      numBoxes: modelInfo.numBoxes,
      boxSize: modelInfo.boxSize,
      numClasses: modelInfo.numClasses,
    };

    if (modelInfo.category === 'SSD') {
      const tensorDic = output.tensor;
      let outputBoxTensor;
      let outputClassScoresTensor;
      if (modelInfo.isQuantized) {
        [outputBoxTensor, outputClassScoresTensor] = deQuantizeOutputTensor(tensorDic.outputBoxTensor,
                                                       tensorDic.outputClassScoresTensor,
                                                       tensorDic.deQuantizedOutputBoxTensor,
                                                       tensorDic.deQuantizedOutputClassScoresTensor,
                                                       deQuantizeParams, options);
      } else {
        outputBoxTensor = tensorDic.outputBoxTensor;
        outputClassScoresTensor = tensorDic.outputClassScoresTensor;
      }
      let anchors = generateAnchors({});
      const start = performance.now();
      decodeOutputBoxTensor({}, outputBoxTensor, anchors);
      this._setDecodeTime(performance.now() - start);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({}, outputBoxTensor, outputClassScoresTensor);
      boxesList = cropSSDBox(this._currentInputElement, totalDetections,
                    boxesList, this._runner._currentModelInfo.margin);
      visualize(canvasShowElement, totalDetections, this._currentInputElement,
        boxesList, scoresList, classesList, output.labels);
    } else {
      const start = performance.now();
      let decode_out = decodeYOLOv2({ nb_class: modelInfo.numClasses },
                         output.tensor, modelInfo.anchors);
      this._setDecodeTime(performance.now() - start);
      let boxes = getBoxes(decode_out, modelInfo.margin);
      drawBoxes(this._currentInputElement, canvasShowElement, boxes, output.labels);
    }
  };
}