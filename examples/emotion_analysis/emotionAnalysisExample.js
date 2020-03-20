class emotionAnalysisExample extends baseCameraExample {
  constructor(models) {
    super(models);
    this._coRunner = null;
    this._currentCoModelInfo = {};
    this._strokedRects = [];
    this._keyPoints = [];
    this._totalInferenceTime = 0.0;
  }

  _setCoModelInfo = (modelInfo) => {
    this._currentCoModelInfo = modelInfo;
  };

  _readyCustomUI = () => {
    $('#fullscreen i svg').click(() => {
      $('#canvasshow').toggleClass('fullscreen');
    });
  };

  _freeMemoryResources = () => {
    if (this._runner) {
      this._runner.deleteAll();
    }

    if (this.corRunner) {
      this._coRunner.deleteAll();
    }
  };

  _getRunner = () => {
    // Overwrite by inherited when example has co-work runners
    if (this._runner == null) {
      this._runner = new faceDetectorRunner();
      this._runner.setProgressHandle(updateLoadingProgressComponent);
    }

    if (this._coRunner == null) {
      this._coRunner = new baseRunner();
      this._coRunner.setProgressHandle(updateLoadingProgressComponent);
    }
  };

  _initRunnerAsync = async () => {
    let currentFDModelId = null;
    let currentEAModelId = null;
    let fdModelList = this._inferenceModels.faceDetection;
    let eaModelList = this._inferenceModels.emotionAnalysis;

    let currentModelArray = null;

    if (this._currentModelId.includes('+')) {
      currentModelArray = this._currentModelId.split('+');
    } else if (this._currentModelId.includes(' ')) {
      currentModelArray = this._currentModelId.split(' ');
    }

    const modelId = currentModelArray[0];

    // Decide whether modleId is currentFDModelId, or is currentEAModelId
    for (let model of fdModelList) {
      if (modelId === model.modelId) {
        currentFDModelId = modelId;
        break;
      }
    }

    if (currentFDModelId != null) {
      currentEAModelId = currentModelArray[1];
    } else {
      currentEAModelId = modelId;
      currentFDModelId = currentModelArray[1];
    }

    const modelInfo = getModelById(fdModelList, currentFDModelId);
    this._setModelInfo(modelInfo);
    await this._runner.initAsync(modelInfo);

    const coModelInfo = getModelById(eaModelList, currentEAModelId);
    this._setCoModelInfo(coModelInfo);
    await this._coRunner.initAsync(coModelInfo);
  };

  _setRunnerModelAsync = async () => {
    await this._runner.initModelAsync(this._currentBackend, this._currentPrefer);
    await this._coRunner.initModelAsync(this._currentBackend, this._currentPrefer);
  };

  _getRequiredOps = () => {
    const fdRequiredOps = this._runner.getRequiredOps();
    const emRequiredOps = this._coRunner.getRequiredOps();
    const requiredOps = new Set([...fdRequiredOps, ...emRequiredOps]);
    return requiredOps;
  };

  _getSubgraphsSummary = () => {
    const fdSummary = this._runner.getSubgraphsSummary();
    const emSummary = this._coRunner.getSubgraphsSummary();
    return fdSummary.concat(emSummary);
  };

  _predictAsync = async () => {
    this._strokedRects = [];
    this._keyPoints = [];
    this._totalInferenceTime = 0.0;
    const fdDrawOptions = {
      inputSize: this._currentModelInfo.inputSize,
      preOptions: this._currentModelInfo.preOptions,
      imageChannels: 4,
    };
    await this._runner.runAsync(this._currentInputElement, fdDrawOptions);
    const fdOutput = this._runner.getOutput();
    this._totalInferenceTime += parseFloat(fdOutput.inferenceTime);
    const height = this._currentInputElement.height
    const width = this._currentInputElement.width;

    if (this._currentModelInfo.category === 'SSD') {
      let anchors = generateAnchors({});
      decodeOutputBoxTensor({}, fdOutput.outputBoxTensor, anchors);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({ num_classes: 2 }, fdOutput.outputBoxTensor, fdOutput.outputClassScoresTensor);
      boxesList = cropSSDBox(this._currentInputElement, totalDetections, boxesList, this._currentModelInfo.margin);
      for (let i = 0; i < totalDetections; ++i) {
        let [ymin, xmin, ymax, xmax] = boxesList[i];
        ymin = Math.max(0, ymin) * height;
        xmin = Math.max(0, xmin) * width;
        ymax = Math.min(1, ymax) * height;
        xmax = Math.min(1, xmax) * width;
        const prob = 1 / (1 + Math.exp(-scoresList[i]));
        const rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
        this._strokedRects.push(rect);
        const drawOptions = {
          inputSize: this._currentCoModelInfo.inputSize,
          preOptions: this._currentCoModelInfo.preOptions,
          imageChannels: 4,
          drawOptions: {
            sx: xmin,
            sy: ymin,
            sWidth: rect[2],
            sHeight: rect[3],
            dWidth: this._currentCoModelInfo.inputSize[1],
            dHeight: this._currentCoModelInfo.inputSize[0],
          },
        };
        await this._coRunner.runAsync(this._currentInputElement, drawOptions);
        let emOutput = this._coRunner.getOutput();
        this._totalInferenceTime += parseFloat(emOutput.inferenceTime);
        this._keyPoints.push(emOutput.outputTensor.slice());
      }
    } else {
      let decode_out = decodeYOLOv2({ nb_class: 1 }, fdOutput.outputTensor, this._currentModelInfo.anchors);
      let outputBoxes = getBoxes(decode_out, this._currentModelInfo.margin);
      for (let i = 0; i < outputBoxes.length; ++i) {
        let [xmin, xmax, ymin, ymax, prob] = outputBoxes[i].slice(1, 6);
        xmin = Math.max(0, xmin) * width;
        xmax = Math.min(1, xmax) * width;
        ymin = Math.max(0, ymin) * height;
        ymax = Math.min(1, ymax) * height;
        let rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
        this._strokedRects.push(rect);
        const drawOptions = {
          inputSize: this._currentCoModelInfo.inputSize,
          preOptions: this._currentCoModelInfo.preOptions,
          imageChannels: 4,
          drawOptions: {
            sx: xmin,
            sy: ymin,
            sWidth: rect[2],
            sHeight: rect[3],
            dWidth: this._currentCoModelInfo.inputSize[1],
            dHeight: this._currentCoModelInfo.inputSize[0],
          },
        };
        await this._coRunner.runAsync(this._currentInputElement, drawOptions);
        let emOutput = this._coRunner.getOutput();
        this._totalInferenceTime += parseFloat(emOutput.inferenceTime);
        this._keyPoints.push(emOutput.outputTensor.slice());
      }
    }

    this._processOutput();
  };

  _processCustomOutput = () => {
    // show inference result
    let texts = [];

    for (let keyPoint of this._keyPoints) {
      let c = getTopClasses(keyPoint, this._currentCoModelInfo.labels, 1);
      texts.push(`${c[0].label}:${c[0].prob}`);
    }

    const canvasShowElement = document.getElementById('canvasshow');
    drawFaceRectangles(this._currentInputElement, canvasShowElement, this._strokedRects, texts);
  };
}