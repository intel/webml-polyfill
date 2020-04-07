class faceRecognitionExample extends baseCameraExample {
  constructor(models) {
    super(models);
    this._coRunner = null;
    this._currentCoModelInfo = {};
    //  { inferenceTime: 0.0, embeddings: [], faceRectangles: [] };
    this._targetEmbeddings = null;
    this._searchEmbeddings = null;
    this._targetEmbeddingsCamera = null;
    this._searchEmbeddingsCamera = null;
    // bDrew: false is for needing draw target image output when using camera
    this._bDrew = false;
    this._totalInferenceTime = 0.0;
  }

  _setCoModelInfo = (modelInfo) => {
    this._currentCoModelInfo = modelInfo;
  };

  _setTargetEmbeddings = (e) => {
    this._targetEmbeddings = e;
  };

  _setSearchEmbeddings = (e) => {
    this._searchEmbeddings = e;
  };

  _setTargetEmbeddingsCamera = (e) => {
    this._targetEmbeddingsCamera = e;
  };

  _setSearchEmbeddingsCamera = (e) => {
    this._searchEmbeddingsCamera = e;
  };

  _setDrewFlag = (flag) => {
    this._bDrew = flag;
  };

  _customUI = () => {
    const targetImageElement = document.getElementById('targetImage');
    const cameraImageElement = document.getElementById('cameraImage');
    const inputElement = document.getElementById('input');
    const targetInputElement = document.getElementById('targetInput');
    const cameraInputElement = document.getElementById('cameraImageInput');

    $('#fullscreen i svg').click(() => {
      $('#cameraShow').toggleClass('fullscreen');
    });

    targetInputElement.addEventListener('change', (e) => {
      let files = e.target.files;

      if (files.length > 0) {
        targetImageElement.src = URL.createObjectURL(files[0]);
        this._setTargetEmbeddings(null);
      }
    }, false);

    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;

      if (files.length > 0) {
        this._setSearchEmbeddings(null);
      }
    }, false);

    cameraInputElement.addEventListener('change', (e) => {
      let files = e.target.files;

      if (files.length > 0) {
        cameraImageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    targetImageElement.addEventListener('load', () => {
      this.main();
    }, false);

    cameraImageElement.addEventListener('load', () => {
      this._setTargetEmbeddingsCamera(null);
      this._setDrewFlag(false);
      this.main();
    }, false);

    $('#targetInput').hide();
    $('#input').hide();
    $('#cameraImageInput').hide();
    $('#feedMediaElement').hide();

    $('#img').click(() => {
      this._setDrewFlag(false);
    });

    $('#backendswitch').click(() => {
      this._setDrewFlag(false);
      this._setTargetEmbeddings(null);
      this._setSearchEmbeddings(null);
      this._setTargetEmbeddingsCamera(null);
      this._setSearchEmbeddingsCamera(null);
    });

    $('input:radio[name=m]').click(() => {
      this._setDrewFlag(false);
      this._setTargetEmbeddings(null);
      this._setSearchEmbeddings(null);
      this._setTargetEmbeddingsCamera(null);
      this._setSearchEmbeddingsCamera(null);
    });

    $('input:radio[name=bp]').click(() => {
      this._setDrewFlag(false);
      this._setTargetEmbeddings(null);
      this._setSearchEmbeddings(null);
      this._setTargetEmbeddingsCamera(null);
      this._setSearchEmbeddingsCamera(null);
    });

    $('input:radio[name=bw]').click(() => {
      this._setDrewFlag(false);
      this._setTargetEmbeddings(null);
      this._setSearchEmbeddings(null);
      this._setTargetEmbeddingsCamera(null);
      this._setSearchEmbeddingsCamera(null);
    });
  };

  _freeMemoryResources = () => {
    if (this._runner) {
      this._runner.deleteAll();
    }

    if (this._coRunner) {
      this._coRunner.deleteAll();
    }
  };

  _getRunner = () => {
    if (this._runner == null) {
      this._runner = new faceDetectorRunner();
      this._runner.setProgressHandler(updateLoadingProgressComponent);
    }

    if (this._coRunner == null) {
      this._coRunner = new baseRunner();
      this._coRunner.setProgressHandler(updateLoadingProgressComponent);
    }
  };

  _loadModel = async () => {
    let currentFDModelId = null;
    let currentFRModelId = null;
    let fdModelList = this._inferenceModels.faceDetection;
    let frModelList = this._inferenceModels.faceRecognition;

    let currentModelArray = null;

    if (this._currentModelId.includes('+')) {
      currentModelArray = this._currentModelId.split('+');
    } else if (this._currentModelId.includes(' ')) {
      currentModelArray = this._currentModelId.split(' ');
    }

    const modelId = currentModelArray[0];

    // Decide whether modleId is currentFDModelId, or is currentFRModelId
    for (let model of fdModelList) {
      if (modelId === model.modelId) {
        currentFDModelId = modelId;
        break;
      }
    }

    if (currentFDModelId != null) {
      currentFRModelId = currentModelArray[1];
    } else {
      currentFRModelId = modelId;
      currentFDModelId = currentModelArray[1];
    }

    const modelInfo = getModelById(fdModelList, currentFDModelId);
    this._setModelInfo(modelInfo);
    await this._runner.loadModel(modelInfo);

    const coModelInfo = getModelById(frModelList, currentFRModelId);
    this._setCoModelInfo(coModelInfo);
    await this._coRunner.loadModel(coModelInfo);
  };

  _setSupportedOps = (ops) => {
    this._runner.setSupportedOps(ops);
    this._coRunner.setSupportedOps(ops);
  };

  _compileModel = async () => {
    await this._runner.compileModel(this._currentBackend, this._currentPrefer);
    await this._coRunner.compileModel(this._currentBackend, this._currentPrefer);
  };

  _getRequiredOps = () => {
    const fdRequiredOps = this._runner.getRequiredOps();
    const frRequiredOps = this._coRunner.getRequiredOps();
    const requiredOps = new Set([...fdRequiredOps, ...frRequiredOps]);
    return requiredOps;
  };

  _getSubgraphsSummary = () => {
    const fdSummary = this._runner.getSubgraphsSummary();
    const frSummary = this._coRunner.getSubgraphsSummary();
    return fdSummary.concat(frSummary);
  };

  _getEmbeddings = async (element) => {
    let inferenceTime = 0.0;
    let strokedRects = [];
    let embeddings = [];
    const fdDrawOptions = {
      inputSize: this._currentModelInfo.inputSize,
      preOptions: this._currentModelInfo.preOptions,
      imageChannels: 4,
    };
    let fdOutput = null;
    let frOutput = null;
    await this._runner.run(element, fdDrawOptions);
    fdOutput = this._runner.getOutput();
    inferenceTime += parseFloat(fdOutput.inferenceTime);
    const height = element.height
    const width = element.width;
    if (this._currentModelInfo.category === 'SSD') {
      let anchors = generateAnchors({});
      decodeOutputBoxTensor({}, fdOutput.outputBoxTensor, anchors);
      let [totalDetections, boxesList, scoresList, classesList] = NMS({ num_classes: 2 }, fdOutput.outputBoxTensor, fdOutput.outputClassScoresTensor);
      boxesList = cropSSDBox(element, totalDetections, boxesList, this._currentModelInfo.margin);
      for (let i = 0; i < totalDetections; ++i) {
        let [ymin, xmin, ymax, xmax] = boxesList[i];
        ymin = Math.max(0, ymin) * height;
        xmin = Math.max(0, xmin) * width;
        ymax = Math.min(1, ymax) * height;
        xmax = Math.min(1, xmax) * width;
        const prob = 1 / (1 + Math.exp(-scoresList[i]));
        const rect = [xmin, ymin, xmax - xmin, ymax - ymin, prob];
        strokedRects.push(rect);
        const frDrawOptions = {
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
        await this._coRunner.run(element, frDrawOptions);
        frOutput = this._coRunner.getOutput();
        inferenceTime += parseFloat(frOutput.inferenceTime);
        let [...normEmbedding] = Float32Array.from(frOutput.outputTensor);
        embeddings.push(normEmbedding);
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
        strokedRects.push(rect);
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
        await this._coRunner.run(this._currentInputElement, drawOptions);
        let frOutput = this._coRunner.getOutput();
        inferenceTime += parseFloat(frOutput.inferenceTime);
        let [...normEmbedding] = Float32Array.from(frOutput.outputTensor);
        embeddings.push(normEmbedding);
      }
    }

    return {inferenceTime: inferenceTime,
            strokedRects: strokedRects,
            embeddings: embeddings,};

  };

  _predict = async () => {
    if (this._currentInputType === 'image') {
      let flag1 = false;
      let flag2 = false;
      if (this._targetEmbeddings == null) {
        const targetImageElement = document.getElementById('targetImage');
        const embeddings = await this._getEmbeddings(targetImageElement);
        this._setTargetEmbeddings(embeddings);
        flag1 = true;
      }
      if (this._searchEmbeddings == null) {
        const sEmbeddings = await this._getEmbeddings(this._feedElement);
        this._setSearchEmbeddings(sEmbeddings);
        flag2 = true;
      }
      if (flag1 && flag2) {
        this._totalInferenceTime = this._targetEmbeddings.inferenceTime + this._searchEmbeddings.inferenceTime;
      } else if (flag1 && !flag2) {
        this._totalInferenceTime = this._targetEmbeddings.inferenceTime;
      } else if (!flag1 && !flag2) {
        this._totalInferenceTime = this._searchEmbeddings.inferenceTime;
      }

    } else if (this._currentInputType === 'camera') {
      const scEmbeddings = await this._getEmbeddings(this._currentInputElement);
      this._setSearchEmbeddingsCamera(scEmbeddings);
      if (this._targetEmbeddingsCamera == null) {
        const cameraImageElement = document.getElementById('cameraImage');
        const cEmbeddings = await this._getEmbeddings(cameraImageElement);
        this._setTargetEmbeddingsCamera(cEmbeddings);
        this._totalInferenceTime = this._targetEmbeddingsCamera.inferenceTime + this._searchEmbeddingsCamera.inferenceTime;
      } else {
        this._totalInferenceTime = this._searchEmbeddingsCamera.inferenceTime;
      }
    }

    this._processOutput();
  };

  _processCustomOutput = () => {
    const supportedOps = getSupportedOps(this._currentBackend, this._currentPrefer);

    if (this._currentInputType === 'image') {
      let targetTextClasses = [];
      for (let i in this._targetEmbeddings.embeddings) {
        targetTextClasses.push(parseInt(i) + 1);
      }
      // show inference result
      const targetImageElement = document.getElementById('targetImage');
      const targetCanvasShowElement = document.getElementById('targetCanvasShow');
      drawFaceRectangles(targetImageElement,
                         targetCanvasShowElement,
                         this._targetEmbeddings.strokedRects,
                         targetTextClasses, 300);
      const searchCanvasShowElement = document.getElementById('searchCanvasShow');
      const searchTextClasses = getFRClass(this._targetEmbeddings.embeddings,
                                           this._searchEmbeddings.embeddings,
                                           this._currentCoModelInfo.postOptions);
      drawFaceRectangles(this._feedElement,
                         searchCanvasShowElement,
                         this._searchEmbeddings.strokedRects,
                         searchTextClasses, 300);
    } else if (this._currentInputType === 'camera') {
      if (!this._bDrew) {
        let targetTextClassesCamera = [];
        for (let i in this._targetEmbeddingsCamera.embeddings) {
          targetTextClassesCamera.push(parseInt(i) + 1);
        }
        // show inference result
        const cameraImageElement = document.getElementById('cameraImage');
        const cameraImageShowElement = document.getElementById('cameraImageShow');
        drawFaceRectangles(cameraImageElement,
                          cameraImageShowElement,
                          this._targetEmbeddingsCamera.strokedRects,
                          targetTextClassesCamera, 300);
          this._setDrewFlag(true);
      }
      const cameraShowElement = document.getElementById('cameraShow');
      const searchTextClassesCamera = getFRClass(this._targetEmbeddingsCamera.embeddings,
                                           this._searchEmbeddingsCamera.embeddings,
                                           this._currentCoModelInfo.postOptions);
      drawFaceRectangles(this._feedMediaElement,
                         cameraShowElement,
                         this._searchEmbeddingsCamera.strokedRects,
                         searchTextClassesCamera, 300);
    }
  };
}
