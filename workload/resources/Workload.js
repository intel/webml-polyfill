const SUPPORTED_WORKLOAD_CATG = {
  // use single model
  'Image Classification': {model: imageClassificationModels},
  'Object Detection': {model: objectDetectionModels},
  'Semantic Segmentation': {model: semanticSegmentationModels},
  'Skeleton Detection': {model: humanPoseEstimationModels},
  'Super Resolution': {model: superResolutionModels},
  // use dual models
  'Emotion Analysis': {
    faceDetection: faceDetectionModels,
    emotionAnalysis: emotionAnalysisModels,
  },
  'Facial Landmark Detection': {
    faceDetection: faceDetectionModels,
    facialLandmarkDetection: facialLandmarkDetectionModels,
  }
};

const DEFAULT_FRAMEWORK = 'WebNN';

class Workload {
  constructor() {
    this._imageElement = document.getElementById('image');
    this._inputElement = document.getElementById('input');
    this._showCanvasElement = document.getElementById('showcanvas');
    this._outCtx = this._showCanvasElement.getContext("2d");
    this._pickBtnElement = document.getElementById('pickbutton');
    this._runButton = document.getElementById('runbutton');

    this._currentFramework;
    this._currentCategory;
    this._currentModelId;
    this._currentCoModelId = null;

    // only for WebNN
    this._currentBackend; // 'WASM' | 'WebGL' | 'WebNN'
    this._currentPrefer;

    // only for 'OpenCV.js'
    this._currentOpenCVjsBackend; // 'WASM' | 'SIMD' | 'Threads' | 'Threads+SIMD'
    this._runtimeInitialized = false; // for 'OpenCV.js', always true for other framework

    this._executor;
  }

  _setFramework = (framework) => {
    this._currentFramework = framework;
    console.log(`Current Framework: ${framework}`);
  };

  _setCategory = (category) => {
    this._currentCategory = category;
  };

  _setModelId = (modelId) => {
    this._currentModelId = modelId;
    console.log(`Current ModelId: ${modelId}`);
  };

  _setCoModelId = (modelId) => {
    this._currentCoModelId = modelId;
    console.log(`Current coModelId: ${modelId}`);
  };

  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo;
  };

  _setCoModelInfo = (modelInfo) => {
    this._currentCoModelInfo = modelInfo;
  };

  _setBackend = (backend) => {
    this._currentBackend = backend;
    console.log(`Current Backend: ${backend}`);
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
    console.log(`Current Prefer: ${prefer}`);
  };

  /**
   * This method is to set '_currentOpenCVjsBackend'.
   * @param {string} backend A string that for OpenCV.js backend.
   */
  _setOpenCVjsBackend = (backend) => {
    this._currentOpenCVjsBackend = backend;
    console.log(`Current OpenCVjs Backend: ${backend}`);
  };

  /**
   * This method is to set '_runtimeInitialized'.
   * @param {boolean} flag A boolean that for whether OpenCV.js runtime initialized.
   */
  _setRuntimeInitialized = (flag) => {
    this._runtimeInitialized = flag;
  };

  _reset = () => {
    // for clearing inference labels of Image Classification example or Semantic Segmentation example
    $('.labels-wrapper').hide();

    this._outCtx.clearRect(0, 0, this._showCanvasElement.width, this._showCanvasElement.height);
    this._outCtx.drawImage(this._imageElement, 0, 0);
  };

  _setCategoryComponent = (categoryValue = null) => {
    const getWorkloadCategory = (framework) => {
      let catg = [];
      switch (framework) {
        case 'WebNN':
          catg = Object.keys(SUPPORTED_WORKLOAD_CATG);
          break;
        case 'OpenCV.js':
          for (let c in SUPPORTED_WORKLOAD_CATG) {
            for (const modelsList of Object.values(SUPPORTED_WORKLOAD_CATG[c])) {
              for (const model of modelsList) {
                if (model.framework && model.framework.includes(framework)) {
                  catg.push(c);
                }
              }
            }
          }
          catg = [...new Set(catg)];
          break;
        default:
          // nerver goes here
      };
      return catg;
    };

    $('#categoryselect').empty();
    const categoryList = getWorkloadCategory(this._currentFramework);
    this._setCategory(categoryList[0]);
    for (let category of categoryList) {
      let option = document.createElement('option');
      option.textContent = category;
      document.querySelector('#categoryselect').appendChild(option);
    }
    categoryValue = categoryValue === null ? categoryList[0] : categoryValue;
    this._updateSelectBox('categoryselect', categoryValue);
  };

  _setModelComponent = (modelId = null, coModelId = null) => {
    const setModelOptions = (models, framework, selectorId, modelId) => {
      for (let model of models) {
        let option = document.createElement('option');
        if (framework === 'OpenCV.js') {
          if (model.framework && model.framework.includes(this._currentFramework)) {
            option.value = model.modelId;
            option.textContent = model.modelName;
            document.querySelector('#' + selectorId).appendChild(option);
          }
        } else {
          option.value = model.modelId;
          option.textContent = model.modelName;
          document.querySelector('#' + selectorId).appendChild(option);
        }
      }
      if (modelId != null) {
        this._updateSelectBox(selectorId, modelId);
      } else {
        this._updateSelectBox(selectorId, models[0].modelId);
      }
    };

    $('#modellabel1').hide();
    $('#modelselect1').empty();
    $('#modellabel2').hide();
    $('#modelselect2').empty();
    $('#modelselect2').hide();

    const modelsDict = SUPPORTED_WORKLOAD_CATG[this._currentCategory];
    if (Object.keys(modelsDict).length === 1) {
      setModelOptions(modelsDict.model, this._currentFramework, 'modelselect1', modelId);
    } else {
      let index = 1;
      let selectedModelId = modelId;
      for (let k in modelsDict) {
        $('#modellabel' + index).html(k.charAt(0).toUpperCase() + k.substring(1, k.length));
        setModelOptions(modelsDict[k], this._currentFramework, 'modelselect' + index, selectedModelId);
        index += 1;
        selectedModelId = coModelId;
      }
      $('#modellabel1').show();
      $('#modellabel2').show();
      $('#modelselect2').show();
    }

    this._initModelId();
  };

  _initModelId = () => {
    this._setModelId($('#modelselect1').val());

    const coModelId = $('#modelselect2').val();
    if (coModelId != null) {
      this._setCoModelId(coModelId);
    }
  };

  _modelChangeBinding = () => {
    const modelsSelect = document.querySelector('#modelselect1');
    modelsSelect.addEventListener('change', (e) => {
      this._setModelId(e.target.value);
      this._reset();
      this._updateSelectBox('modelselect1', e.target.value);
    }, false);
    const coModelsSelect = document.querySelector('#modelselect2');
    coModelsSelect.addEventListener('change', (e) => {
      this._setCoModelId(e.target.value);
      this._updateSelectBox('modelselect2', e.target.value);
    }, false);
  };

  _inputChangeBinding = () => {
    this._inputElement.addEventListener('change', (e) => {
      this._reset();
      let files = e.target.files;
      if (files.length > 0) {
        this._imageElement.src = URL.createObjectURL(files[0]);
      }
      this._setImageSrc();
    }, false);
  };

  _setWebNNComponents = () => {
    $('#opencvjscatg').hide();
    $('#opencvjsbackend').hide();
    $('#webnncatg').show();
    this._setCategoryComponent();
    this._setRuntimeInitialized(true);
    this._setBackend('WASM');
    this._setPrefer('none');
    this._setModelComponent();
    this._modelChangeBinding();
    this._inputChangeBinding();
    $('#webnnbackend').show();
    $('#webnnprefer').show();
    $("#preferselect option[value='none']").prop("selected", true);
    document.querySelector('#preferselect option[value=none]').disabled = false;
    this._updateOpsSelect();
    this._showImage();
  };

  _setOpenCVjsComponents = (category, modelId, OpenCVjsBackend) => {
    $('#webnncatg').hide();
    $('#webnnbackend').hide();
    $('#webnnprefer').hide();
    $('#supportedopsselect').hide();
    $('#opencvjscatg').show();
    this._setRuntimeInitialized(false);
    this._setCategoryComponent(category);
    this._setModelComponent(modelId);
    OpenCVjsBackend = typeof OpenCVjsBackend === 'undefined' ? 'WASM' : OpenCVjsBackend;
    this._updateSelectBox('opencvjsbackend', OpenCVjsBackend);
    this._setOpenCVjsBackend(OpenCVjsBackend);
    this._modelChangeBinding();
    this._inputChangeBinding();
    $('#opencvjsbackend').show();
    this._showImage();
  };

  _showImage = () => {
    this._showCanvasElement.setAttribute("width", this._imageElement.naturalWidth);
    this._showCanvasElement.setAttribute("height", this._imageElement.naturalHeight);
    this._outCtx.drawImage(this._imageElement, 0, 0);
  };

  _setImageSrc = (category) => {
    if (typeof category === 'undefined') {
      const inputFile = document.getElementById('input').files[0];
      this._imageElement.src = URL.createObjectURL(inputFile);
    } else {
      switch (this._currentCategory) {
        case 'Image Classification':
          this._imageElement.src = '../examples/image_classification/img/test.jpg';
          break;
        case 'Object Detection':
          this._imageElement.src = '../examples/object_detection/img/image1.jpg';
          break;
        case 'Semantic Segmentation':
          this._imageElement.src = '../examples/semantic_segmentation/img/woman.jpg';
          break;
        case 'Skeleton Detection':
          this._imageElement.src = '../examples/skeleton_detection/img/download.png';
          break;
        case 'Super Resolution':
          this._imageElement.src = '../examples/super_resolution/img/mushroom.png';
          break;
        case 'Emotion Analysis':
          this._imageElement.src = '../examples/emotion_analysis/img/image1.jpg';
          break;
        case 'Facial Landmark Detection':
          this._imageElement.src = '../examples/facial_landmark_detection/img/image1.jpg';
          break;
        default:
          this._imageElement.src = '../examples/image_classification/img/test.jpg';
      }
    }

    this._imageElement.onload = () => {
      this._showImage();
    };
  };

  _updateOpsSelect = () => {
    if (this._currentBackend !== 'WebNN' && this._currentPrefer !== 'none') {
      // hybrid mode
      $('#supportedopsselect').show();
    } else {
      // solo mode
      $('#supportedopsselect').hide();
    }
  };

  _setCustomComponents = (category, modelId, OpenCVjsBackend) => {
    switch (this._currentFramework) {
      case 'WebNN':
        this._setWebNNComponents();
        break;
      case 'OpenCV.js':
        this._setOpenCVjsComponents(category, modelId, OpenCVjsBackend);
        break;
      default:
        // nerver goes here
    }
  };

  _updateSelectBox = (selectElementId, value) => {
    const selectElement = document.getElementById(selectElementId);
    let options = selectElement.options;

    for(let i = 0; i < options.length; i++) {
      if( options[i].value === value) {
        options[i].defaultSelected = true;
        options[i].selected = true;
      } else {
        options[i].defaultSelected = false;
        options[i].selected = false;
      }
    }
  };

  _storageOpenCVjsEnv = () => {
    window.sessionStorage.setItem("framework", this._currentFramework);
    window.sessionStorage.setItem("category", this._currentCategory);
    window.sessionStorage.setItem("modelId", this._currentModelId);
    window.sessionStorage.setItem("backend", this._currentOpenCVjsBackend);
  };

  /**
   * This method is to prepare components on UI and set some click trigger bindings.
   * Compoents including:
   *  frameworks, models, backend, progress, inference result, etc. components
   * Click trigger bindings:
   *  framework click trigger bindings
   *  model element click trigger bindings
   *  backend element click trigger bindings
   */
  UI = () => {
    let frameworks = [];
    for (let c in SUPPORTED_WORKLOAD_CATG) {
      let framworkList = getFrameworkList(SUPPORTED_WORKLOAD_CATG[c]);
      frameworks = frameworks.concat(framworkList);
    }

    frameworks = [...new Set(frameworks)];
    for (let framework of frameworks) {
      let option = document.createElement('option');
      option.textContent = framework;
      option.id = framework.replace('.', '');
      document.querySelector('#frameworkselect').appendChild(option);
    }

    let currentFramework = window.sessionStorage.getItem("framework");
    let currentCategory = window.sessionStorage.getItem("category");
    let currentModelId = window.sessionStorage.getItem("modelId");
    let currentOpenCVjsBackend = window.sessionStorage.getItem("backend");
    window.sessionStorage.clear();

    currentFramework = currentFramework === null ? DEFAULT_FRAMEWORK : currentFramework;
    this._updateSelectBox('frameworkselect', currentFramework);
    this._setFramework(currentFramework);
    this._setCustomComponents(currentCategory, currentModelId, currentOpenCVjsBackend);

    // framework trigger
    const frameworkSelect = document.querySelector('#frameworkselect');
    frameworkSelect.addEventListener('change', (e) => {
      let framework = e.target.value;
      this._updateSelectBox('frameworkselect', framework);
      this._setFramework(framework);
      this._setCustomComponents();
      this._setImageSrc(this._currentCategory);
      this._reset();
      this._getExecutor();
    }, false);

    // category trigger
    const categorySelect = document.querySelector('#categoryselect');
    categorySelect.addEventListener('change', (e) => {
      let category = e.target.value;
      this._updateSelectBox('categoryselect', category);
      this._setCategory(category);
      this._setCoModelId(null);
      this._setModelComponent();
      this._setImageSrc(category);
      this._reset();
      this._getExecutor();
    }, false);

    let backend;
    // backend trigger / WebNN
    const webnnBackendSelect = document.querySelector('#webnnbackend');
    webnnBackendSelect.addEventListener('change', (e) => {
      backend = e.target.value;
      this._reset();
      this._updateSelectBox('webnnbackend', backend);
      this._setBackend(backend);
      if (backend === 'WebNN') {
        this._updateSelectBox('preferselect', 'sustained');
        document.querySelector('#preferselect option[value=none]').disabled = true;
        this._setPrefer('sustained');
      } else {
        this._updateSelectBox('preferselect', 'none');
        this._setPrefer('none');
      }
      // hiden ops select for WebNN backend
      this._updateOpsSelect();
    }, false);

    // backend trigger / OpenCV.js
    const opencvjsBackendSelect = document.querySelector('#opencvjsbackend');
    opencvjsBackendSelect.addEventListener('change', (e) => {
      backend = e.target.value;
      this._reset();
      this._updateSelectBox('opencvjsbackend', backend);
      this._setOpenCVjsBackend(backend);
      this._storageOpenCVjsEnv();
      window.location.reload(true);
    }, false);

    // prefere trigger  / WebNN
    const preferSelect = document.querySelector('#preferselect');
    preferSelect.addEventListener('change', (e) => {
      let prefer = e.target.value;
      this._reset();
      this._updateSelectBox('preferselect', prefer);
      this._setPrefer(prefer);
      // show ops select for hybrid mode
      this._updateOpsSelect();
    }, false);

    const selectAllOpsElement = document.getElementById('selectallops');
    selectAllOpsElement.addEventListener('click', () => {
      document.querySelectorAll('input[name=supportedop]').forEach((x) => {
        x.checked = true;
      });
    }, false);

    const uncheckAllOpsElement = document.getElementById('uncheckallops');
    uncheckAllOpsElement.addEventListener('click', () => {
      document.querySelectorAll('input[name=supportedop]').forEach((x) => {
        x.checked = false;
      });
    }, false);

    $('.labels-wrapper').hide();

    this._runButton.addEventListener('click', () => {
      this._reset();
      this._main();
    });

    this._getExecutor();
  };

  _getExecutor = () => { //todo
    switch (this._currentCategory) {
      case 'Image Classification':
        if (this._currentFramework === 'WebNN') {
          this._executor = new ImageClassificationWebNNExecutor();
        } else if (this._currentFramework === 'OpenCV.js') {
          this._executor = new ImageClassificationOpenCVExecutor();
        }
        break;
      case 'Object Detection':
        this._executor = new ObjectDetectionWebNNExecutor();
        break;
      case 'Semantic Segmentation':
        this._executor = new SemanticSegmentationWebNNExecutor();
        break;
      case 'Skeleton Detection':
        this._executor = new SkeletonDetectionWebNNExecutor();
        break;
      case 'Super Resolution':
        this._executor = new SuperResolutionWebNNExecutor();
        break;
      case 'Emotion Analysis':
        this._executor = new EmotionAnalysisWebNNExecutor();
        break;
      case 'Facial Landmark Detection':
        this._executor = new FacialLandmarkDetectionWebNNExecutor();
        break;
      default:
        // nerver goes here
    }
  };

  _readyExecutor = async () => {
    this._executor.setInputElement(this._imageElement);
    this._executor.getRunner();
    this._executor.setCategory(this._currentCategory);
    await this._executor.doInitialRunner(this._currentModelId, this._currentCoModelId);

    if (this._currentFramework === 'WebNN') {
      this._executor.setBackend(this._currentBackend);
      this._executor.setPrefer(this._currentPrefer);
      if (this._currentPrefer !== 'none') {
        this._executor.setEagerMode($('#eagermode').prop('checked'));
        let ops = Array.from(document.querySelectorAll('input[name=supportedop]:checked')).map(
                    x => parseInt(x.value));
        this._executor.setSupportedOps(ops);
      }
    } else if (this._currentFramework === 'OpenCV.js') {
      this._executor.setBackend(this._currentOpenCVjsBackend);
    }
    await this._executor.loadAndCompileModel();
  };
  /**
   * This method is for running OpenCV.js framework, execute main method when OpenCV.js runtime was initialized.
   */
  _onOpenCvReady = async () => {
    cv = await cv;
    this._setRuntimeInitialized(true);
    this._main();
  };

  _main = async () => {
    const summarize = (results) => {
      if (results.length !== 0) {
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        let d = results.reduce((d, v) => {
          d.sum += v;
          d.sum2 += v * v;
          return d;
        }, {
          sum: 0,
          sum2: 0
        });
        let mean = d.sum / results.length;
        let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
        return {
          mean: mean,
          std: std
        };
      } else {
        return null;
      }
    };

    // Use median inference metric
    const summarizeMedian = (results) => {
      if (results.length !== 0) {
        // remove first run, which is regarded as "warming up" execution
        results.shift();
        const median = arr => {
          arr = arr.sort((a, b) => a - b);
          return arr.length % 2 !== 0 ? arr[Math.floor(arr.length / 2)] : (arr[arr.length / 2 - 1] + arr[arr.length / 2]) / 2;
        };
        let med = median(results)
        return {
          median: med
        };
      } else {
        return null;
      }
    };

    const summarizeProf = (results) => {
      const lines = [];
      if (!results) {
        return lines;
      }
      lines.push(`Execution calls: ${results.epochs} (omitted ${results.warmUpRuns} warm-up runs)`);
      lines.push(`Supported Ops: ${results.supportedOps.join(', ') || 'None'}`);
      lines.push(`Mode: ${results.mode}`);

      let polyfillTime = 0;
      let webnnTime = 0;
      for (const t of results.timings) {
        lines.push(`${t.elpased.toFixed(8).slice(0, 7)} ms\t- (${t.backend}) ${t.summary}`);
        if (t.backend === 'WebNN') {
          webnnTime += t.elpased;
        } else {
          polyfillTime += t.elpased;
        }
      }
      lines.push(`Polyfill time: ${polyfillTime.toFixed(5)} ms`);
      lines.push(`WebNN time: ${webnnTime.toFixed(5)} ms`);
      lines.push(`Sum: ${(polyfillTime + webnnTime).toFixed(5)} ms`);
      return lines;
    }

    this._inputElement.setAttribute('disabled', true);
    this._pickBtnElement.setAttribute('class', 'btn btn-primary disabled');
    this._runButton.setAttribute('disabled', true);
    let logger = new Logger(document.querySelector('#log'));

    try {
      if (this._currentFramework === 'OpenCV.js') {
        if (!this._runtimeInitialized) {
          console.log(`Runtime isn't initialized`);
          asyncLoadScript(`../examples/util/opencv.js/${this._currentOpenCVjsBackend}/opencv.js`, this._onOpenCvReady);
          return;
        }
      }
      const iterations = Number($('#iterations').val()) + 1;

      let modelName = $('#modelselect1 option:selected').text();
      let logModelName = modelName;
      let coModelName = $('#modelselect2 option:selected').text();
      if (coModelName !== "") {
        logModelName = `${modelName} + ${coModelName}`;
      }

      let backend;
      if (this._currentFramework === 'WebNN') {
        backend = this._currentBackend;
        let prefer = $('#preferselect option:selected').text();
        if (prefer !== "") {
          if (backend === 'WebNN') {
            backend = `${backend} (Preference: ${prefer})`;
          }
        }
      } else if (this._currentFramework === 'OpenCV.js') {
        backend = this._currentOpenCVjsBackend;
      }

      let configuration = {
        "Framework": this._currentFramework,
        "Category": this._currentCategory,
        "Model Name": logModelName,
        "backend": backend,
        "iterations": iterations,
      };
      logger.group('Benchmark');
      logger.group('Environment Information');
      logger.log(`${'UserAgent'.padStart(12)}: ${(navigator.userAgent) || '(N/A)'}`);
      logger.log(`${'Platform'.padStart(12)}: ${(navigator.platform || '(N/A)')}`);
      logger.groupEnd();

      logger.group('Configuration');
      Object.keys(configuration).forEach(key => {
        logger.log(`${key.padStart(12)}: ${configuration[key]}`);
      });
      logger.groupEnd();

      logger.group('Run');
      await this._readyExecutor();
      await this._executor.execute(iterations, logger);
      logger.groupEnd();

      const inferenceResults = this._executor.getInferenceResults();

      // Profilling is only for WebNN
      if (this._currentFramework === 'WebNN') {
        if (this._currentBackend !== 'WebNN') {
          if (coModelName !== "") {
            const profilingResults1 = summarizeProf(inferenceResults.profiling[0]);
            if (profilingResults1.length !== 0) {
              logger.group(`Profiling (${modelName})`);
              profilingResults1.forEach((line) => logger.log(line));
              logger.groupEnd();
            }
            const profilingResults2 = summarizeProf(inferenceResults.profiling[1]);
            if (profilingResults2.length !== 0) {
              logger.group(`Profiling (${coModelName})`);
              profilingResults2.forEach((line) => logger.log(line));
              logger.groupEnd();
            }
          } else {
            const profilingResults = summarizeProf(inferenceResults.profiling[0]);
            if (profilingResults.length !== 0) {
              logger.group('Profiling');
              profilingResults.forEach((line) => logger.log(line));
              logger.groupEnd();
            }
          }
        }
      }
      logger.group('Result');
      const results = summarize(inferenceResults.inferenceTimeList);
      const resultsMedian = summarizeMedian(inferenceResults.inferenceTimeList);
      logger.log(`Inference Time (Average): <em style="color:green;font-weight:bolder;">${results.mean.toFixed(2)}+-${results.std.toFixed(2)}</em> [ms]`);
      logger.log(`Inference Time (Median): <em style="color:green;font-weight:bolder;">${resultsMedian.median.toFixed(2)}</em> [ms]`);
      if (inferenceResults.decodeTime !== 0.0) {
        logger.log(`Decode Time: <em style="color:green;font-weight:bolder;">${inferenceResults.decodeTime.toFixed(2)}</em> [ms]`);
      }
    } catch (e) {
      logger.error(e);
    }

    logger.groupEnd();

    this._inputElement.removeAttribute('disabled');
    this._pickBtnElement.setAttribute('class', 'btn btn-primary');
    this._runButton.removeAttribute('disabled');
  };
}
