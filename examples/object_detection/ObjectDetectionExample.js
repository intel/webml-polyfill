class ObjectDetectionExample extends BaseCameraExample {
  constructor(models) {
    super(models);
  }

  /** @override */
  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('#canvasshow').toggleClass('fullscreen');
    });
  };

  /** @override */
  _createRunner = () => {
    const runner = new ObjectDetectionRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _processExtra = (output) => {
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    let canvasShowElement = document.getElementById('canvasshow');
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
      decodeOutputBoxTensor({}, outputBoxTensor, anchors);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({}, outputBoxTensor, outputClassScoresTensor);
      boxesList = cropSSDBox(this._currentInputElement, totalDetections,
                    boxesList, this._runner._currentModelInfo.margin);
      visualize(canvasShowElement, totalDetections, this._currentInputElement,
        boxesList, scoresList, classesList, output.labels);
    } else {
      let decode_out = decodeYOLOv2({ nb_class: modelInfo.numClasses },
                         output.tensor, modelInfo.anchors);
      let boxes = getBoxes(decode_out, modelInfo.margin);
      drawBoxes(this._currentInputElement, canvasShowElement, boxes, output.labels);
    }
  };
}