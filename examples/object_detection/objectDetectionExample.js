class objectDetectionExample extends baseCameraExample {
  constructor(models) {
    super(models);
  }

  _readyCustomUI = () => {
    $('#fullscreen i svg').click(() => {
      $('#canvasshow').toggleClass('fullscreen');
    });
  };

  _createRunner = () => {
    const runner = new objectDetectionRunner();
    runner.setProgressHandle(updateLoadingProgressComponent);
    return runner;
  };

  _processCustomOutput = () => {
    const output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    let canvasShowElement = document.getElementById('canvasshow');
    const modelInfo = this._currentModelInfo;
    const options = {
      numBoxes: modelInfo.numBoxes,
      boxSize: modelInfo.boxSize,
      numClasses: modelInfo.numClasses,
    };

    if (modelInfo.category === 'SSD') {
      let outputBoxTensor;
      let outputClassScoresTensor;
      if (modelInfo.isQuantized) {
        [outputBoxTensor, outputClassScoresTensor] = deQuantizeOutputTensor(output.outputBoxTensor,
                                                       output.outputClassScoresTensor,
                                                       output.deQuantizedOutputBoxTensor,
                                                       output.deQuantizedOutputClassScoresTensor,
                                                       deQuantizeParams, options);
      } else {
        outputBoxTensor = output.outputBoxTensor;
        outputClassScoresTensor = output.outputClassScoresTensor;
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
                         output.outputTensor, modelInfo.anchors);
      let boxes = getBoxes(decode_out, modelInfo.margin);
      drawBoxes(this._currentInputElement, canvasShowElement, boxes, output.labels);
    }
  };
}