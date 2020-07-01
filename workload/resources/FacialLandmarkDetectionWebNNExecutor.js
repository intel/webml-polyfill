class FacialLandmarkDetectionWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
    this._currentCoModelInfo;
    this._coRunner;
    this._strokedRects = [];
    this._keyPoints = [];
    this._totalInferenceTime = 0.0;
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

    const coModelInfo = getModelById(facialLandmarkDetectionModels, coModelId);
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
    // show inference result
    const texts = this._strokedRects.map(r => r[4].toFixed(2));
    const canvasShowElement = document.getElementById('showcanvas');
    drawFaceRectangles(this._currentInputElement, canvasShowElement, this._strokedRects, texts);
    drawKeyPoints(this._currentInputElement, canvasShowElement, this._keyPoints, this._strokedRects);
  };

  /** @override */
  _executeSingle = async () => {
    this._strokedRects = [];
    this._keyPoints = [];
    this._totalInferenceTime = 0.0;
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
    this._totalInferenceTime += parseFloat(fdOutput.inferenceTime);
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
        const fldSSDInput = {
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
        await this._coRunner.run(fldSSDInput);
        let fldOutput = this._coRunner.getOutput();
        this._totalInferenceTime += parseFloat(fldOutput.inferenceTime);
        this._keyPoints.push(fldOutput.tensor.slice());
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
        const fldYOLOInput = {
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
        await this._coRunner.run(fldYOLOInput);
        let fldOutput = this._coRunner.getOutput();
        this._totalInferenceTime += parseFloat(fldOutput.inferenceTime);
        this._keyPoints.push(fldOutput.tensor.slice());
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
