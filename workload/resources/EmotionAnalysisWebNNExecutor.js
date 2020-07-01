class EmotionAnalysisWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
    this._currentCoModelInfo;
    this._coRunner;
    this._strokedRects = [];
    this._keyPoints = [];
  }

  _setCoModelInfo = (modelInfo) => {
    this._currentCoModelInfo = modelInfo;
  };

  /** @override */
  getRunner = () => {
    if (this._runner == null) {
      this._runner = new FaceDetectorRunner();
    }

    if (this._coRunner == null) {
      this._coRunner = new WebNNRunner();
    }
  };

  /** @override */
  _loadModel = async (modelId, coModelId) => {
    const modelInfo = getModelById(faceDetectionModels, modelId);
    this._setModelInfo(modelInfo);
    await this._runner.loadModel(modelInfo, 'workload');

    const coModelInfo = getModelById(emotionAnalysisModels, coModelId);
    this._setCoModelInfo(coModelInfo);
    await this._coRunner.loadModel(coModelInfo, 'workload');
  };

  /** @override */
  _compileModel = async () => {
    const options = {
      backend: this._currentBackend.replace('WebNN', 'WebML'),
      prefer: this._currentPrefer
    };
    await this._runner.compileModel(options);
    await this._coRunner.compileModel(options);
  };

  /** @override */
  _postProcess = (data) => {
    let texts = [];

    for (let keyPoint of this._keyPoints) {
      let c = getTopClasses(keyPoint, this._currentCoModelInfo.labels, 1);
      texts.push(`${c[0].label}:${c[0].prob}`);
    }

    const canvasShowElement = document.getElementById('showcanvas');
    drawFaceRectangles(this._currentInputElement, canvasShowElement, this._strokedRects, texts);
  };

  /** @override */
  _executeSingle = async () => {
    this._strokedRects = [];
    this._keyPoints = [];
    let totalInferenceTime = 0.0;
    const fdInput = {
      src: this._currentInputElement,
      options: {
        inputSize: this._currentModelInfo.inputSize,
        preOptions: this._currentModelInfo.preOptions,
        imageChannels: 4,
      },
    };
    await this._runner.run(fdInput);
    const fdOutput = this._runner.getOutput();
    totalInferenceTime += parseFloat(fdOutput.inferenceTime);
    const height = this._currentInputElement.height
    const width = this._currentInputElement.width;

    if (this._currentModelInfo.category === 'SSD') {
      let anchors = generateAnchors({});
      const start = performance.now();
      decodeOutputBoxTensor({}, fdOutput.tensor.outputBoxTensor, anchors);
      this._setDecodeTime(performance.now() - start);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({ num_classes: 2 }, fdOutput.tensor.outputBoxTensor, fdOutput.tensor.outputClassScoresTensor);
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
        const eaSSDInput = {
          src: this._currentInputElement,
          options: {
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
          },
        };
        await this._coRunner.run(eaSSDInput);
        let emOutput = this._coRunner.getOutput();
        totalInferenceTime += parseFloat(emOutput.inferenceTime);
        this._inferenceTimeList.push(totalInferenceTime);
        this._keyPoints.push(emOutput.tensor.slice());
      }
    } else {
      const start = performance.now();
      let decode_out = decodeYOLOv2({ nb_class: 1 }, fdOutput.tensor, this._currentModelInfo.anchors);
      this._setDecodeTime(performance.now() - start);
      let outputBoxes = getBoxes(decode_out, this._currentModelInfo.margin);
      for (let i = 0; i < outputBoxes.length; ++i) {
        let [xmin, xmax, ymin, ymax, prob] = outputBoxes[i].slice(1, 6);
        xmin = Math.max(0, xmin) * width;
        xmax = Math.min(1, xmax) * width;
        ymin = Math.max(0, ymin) * height;
        ymax = Math.min(1, ymax) * height;
        let rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
        this._strokedRects.push(rect);
        const eaYOLOInput = {
          src: this._currentInputElement,
          options: {
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
          },
        };
        await this._coRunner.run(eaYOLOInput);
        let emOutput = this._coRunner.getOutput();
        totalInferenceTime += parseFloat(emOutput.inferenceTime);
        this._inferenceTimeList.push(totalInferenceTime);
        this._keyPoints.push(emOutput.tensor.slice());
      }
    }
  };

  /** @override */
  _getProfilingResults = () => {
    let profilingResults = null;
    if (this._currentBackend !== 'WebNN') {
      profilingResults = [this._runner._model._compilation._preparedModel.dumpProfilingResults(),
                          this._coRunner._model._compilation._preparedModel.dumpProfilingResults()];
    }
    return profilingResults;
  };
}