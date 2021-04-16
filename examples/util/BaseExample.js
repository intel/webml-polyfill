class BaseExample extends BaseApp {
  constructor(models) {
    super(models);
    this._currentInputType = 'image'; // input type: image, camera, audio, microphone
    // <image> or <audio> element
    this._feedElement = document.getElementById('feedElement');
    // <video> or <audio> element when using live Camera or Microphone
    this._feedMediaElement = document.getElementById('feedMediaElement');
    // Backend type: 'WASM' | 'WebGL' | 'WebML' | 'WebGPU'
    this._currentBackend = 'WASM';
    // Prefer type: 'none' | 'fast' | 'sustained' | 'low' | 'ultra_low'
    this._currentPrefer = 'none';
    // track and stats serve for 'VIDEO' | 'AUDIO' input
    this._track = null;
    this._stats = new Stats();
    this._currentModelInfo = {};
    this._currentFramework; //'OpenCV.js' | 'WebNN'
    this._currentOpenCVJSBackend; // 'WASM' | 'SIMD' | 'Threads' | 'Threads+SIMD'
    this._runtimeInitialized = false; // for 'OpenCV.js', always true for other framework
    this._currentTimeoutId = 0;
    this._isStreaming = false; // for inference camera video
    // _hiddenControlsFlag ('0'/'1') is for UI shows/hides model & backend control
    this._hiddenControlsFlag = '0';
  }

  /**
   * This method is to get default input type,
   * for those examples relevants with camera video, the type is 'image',
   * and for those examples relevants with microphone audio, the type is 'audio'.
   * @returns {string} This returns a string for input type, ['image' | 'audio'].
   */
  _getDefaultInputType = () => {};

  /**
   * This method is to get default input media type,
   * for those examples relevants with camera video, the type is 'camera',
   * and for those examples relevants with audio, the type is 'microphone'.
   * @returns {string} This returns a string for input meadia type, ['camera' | 'microphone'].
   */
  _getDefaultInputMediaType = () => {};

  /**
   * This method is to set '_currentInputType'.
   * @param {string} inputType
   */
  _setInputType = (inputType) => {
    this._currentInputType = inputType;
  };

  /**
   * This method is to set '_currentBackend'.
   * @param {string} backend
   */
  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  /**
   * This method is to set '_currentPrefer'.
   * @param {string} prefer
   */
  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  /**
   * This method is to set '_currentFramework'.
   * @param {string} framework
   */
  _setFramework = (framework) => {
    this._currentFramework = framework;
  };

  /**
   * This method is to set '_track'.
   * @param {!MediaStreamTrack} track
   */
  _setTrack = (track) => {
    this._track = track;
  };

  /**
   * This method is to set '_currentModelInfo'.
   * @param {!Object<string, *>} modelInfo An object that for model info which was configed in modeZoo.js.
   *     An example for model info:
   *       modelInfo = {
   *         modelName: {string}, // 'MobileNet v1 (TFLite)'
   *         format: {string}, // 'TFLite'
   *         modelId: {string}, // 'mobilenet_v1_tflite'
   *         modelSize: {string}, // '16.9MB'
   *         inputSize: {!Array<number>}, // [224, 224, 3]
   *         outputSize: {number} // 1001
   *         modelFile: {string}, // '../image_classification/model/mobilenet_v1_1.0_224.tflite'
   *         labelsFile: {string}, // '../image_classification/model/labels1001.txt'
   *         preOptions: {!Obejct<string, *>}, // {mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5],}
   *         intro: {string}, // 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
   *         paperUrl: {string}, // 'https://arxiv.org/pdf/1704.04861.pdf'
   *       };
   */
  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo;
  };

  /**
   * This method is to set '_currentTimeoutId'.
   * @param {number} id
   */
  _setTimeoutId = (id) => {
    this._currentTimeoutId = id;
  };

  /**
   * This method is to clear a timer set with the setTimeout(this._predictFrame, 0) method.
   */
  _clearTimeout = () => {
    if (this._currentTimeoutId !== 0) {
      clearTimeout(this._currentTimeoutId);
      this._setTimeoutId(0);
    }
  };

  /**
   * This method is to set '_isStreaming'.
   * @param {boolean} flag A boolean that for inference camera video.
   */
  _setStreaming = (flag) => {
    this._isStreaming = flag;
  };

  /**
   * This method is to set '_hiddenControlsFlag'.
   * @param {string|undefined} flag
   */
  _setHiddenControlsFlag = (flag) => {
    // flag: '0' display, '1' hidden
    if (typeof flag === 'undefined') {
      if (this._hiddenControlsFlag === '1') {
        this._hiddenControlsFlag = '0';
      } else {
        this._hiddenControlsFlag = '1';
      }
    } else {
      this._hiddenControlsFlag = flag;
    }
  };

  /**
   * This method is to set '_currentOpenCVJSBackend'.
   * @param {string} backend A string that for OpenCV.js backend.
   */
  _setOpenCVJSBackend = (backend) => {
    this._currentOpenCVJSBackend = backend;
  };

  /**
   * This method is to set '_runtimeInitialized'.
   * @param {boolean} flag A boolean that for whether OpenCV.js runtime initialized.
   */
  _setRuntimeInitialized = (flag) => {
    this._runtimeInitialized = flag;
  };

  /**
   * This method is to update History Entry URL.
   * @param {string|undefined} url
   */
  _updateHistoryEntryURL = (url) => {
    let locSearch;
    if (typeof url !== 'undefined') {
      if (url === "") {
        this._setBackend('WASM');
        this._setPrefer('none');
        this._setModelId('none');
        let inputType = this._getDefaultInputType();
        this._setInputType(inputType);
        this._setInputElement(this._feedElement);
        this._setHiddenControlsFlag('0');
        this._setFramework('WebNN');
        this._setRuntimeInitialized(true);
        $('.backend').show();
        $('.opencvjsbackend').hide();
        locSearch = `?prefer=${this._currentPrefer}&b=${this._currentBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
      } else {
        // Parse seach params, and prepare inference env
        let modelId = parseSearchParams('m');
        this._setModelId(modelId);
        let feedEle = null;
        const inputType = parseSearchParams('s');
        this._setInputType(inputType);
        switch (inputType.toLowerCase()) {
          case 'image':
          case 'audio':
            feedEle = this._feedElement;
            break;
          case 'camera':
          case 'microphone':
            feedEle = this._feedMediaElement;
            break;
          default:
            showErrorComponent(`Invalid url`, "Invalide value of 's' parameter of typed url.It requires 'file' or 'stream' .");
            return;
        }
        this._setInputElement(feedEle);
        const flag = parseSearchParams('d');
        this._setHiddenControlsFlag(flag);
        const framework = parseSearchParams('f');
        this._setFramework(framework);
        switch (framework) {
          case 'WebNN':
            $('.backend').show();
            $('.opencvjsbackend').hide();
            this._setRuntimeInitialized(true);
            const prefer = parseSearchParams('prefer');
            this._setPrefer(prefer);
            const backend = parseSearchParams('b');
            this._setBackend(backend);
            locSearch = `?prefer=${this._currentPrefer}&b=${this._currentBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
            break;
          case 'OpenCV.js':
            $('.backend').hide();
            $('.opencvjsbackend').show();
            const opencvBackend = parseSearchParams('b');
            if (opencvBackend != this._currentOpenCVJSBackend) {
              showOpenCVRuntimeProgressComponent();
              asyncLoadScript(`../util/opencv.js/${opencvBackend.replace(' ', '+')}/opencv.js`, this.onOpenCvReady);
              this._setOpenCVJSBackend(opencvBackend);
            }
            locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
            break;
        }
      }
    } else {
      switch (this._currentFramework) {
        case 'WebNN':
          locSearch = `?prefer=${this._currentPrefer}&b=${this._currentBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
          break;
        case 'OpenCV.js':
          locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
          break;
      }
    }
    window.history.pushState(null, null, locSearch);
  };

  /**
   * This method is to show extra UI parts by example besides common UI parts.
   */
  _commonUIExtra = () => {
  };

  /**
   * This method is to show components on UI according to selected framework.
   * For 'WebNN' framework, it will show supported WebNN models, WebNN backends.
   * For 'OpenCV.js' framework, it will show supported OpenCV.js models, OpenCV.js backends.
   */
  _showDynamicComponents = () => {
    // set model components according "Framework"
    const showInferenceModels = selectModelFromGivenModels(this._inferenceModels, this._currentFramework);
    setModelComponents(showInferenceModels, this._currentModelId);
    switch (this._currentFramework) {
      case 'WebNN':
        $('.backend').show();
        // set preference tip components
        setPreferenceTipComponents();
        // set backend components
        if (hasSearchParam('b')) {
          $('.backend input').removeAttr('checked');
          $('.backend label').removeClass('checked');
          $('#' + parseSearchParams('b')).attr('checked', 'checked');
          $('#l-' + parseSearchParams('b')).addClass('checked');
        }
        if (hasSearchParam('prefer')) {
          $('.prefer input').removeAttr('checked');
          $('.prefer label').removeClass('checked');
          $('#' + parseSearchParams('prefer')).attr('checked', 'checked');
          $('#l-' + parseSearchParams('prefer')).addClass('checked');
        }
        updateBackendComponents(this._currentBackend, this._currentPrefer);
        $('.opencvjsbackend').hide();
        break;
      case 'OpenCV.js':
        $('.backend').hide();
        $('.opencvjsbackend').show();
        updateOpenCVJSBackendComponentsStyle(this._currentOpenCVJSBackend);
        break;
    }
    updateSIMDNotes();
  };

  /**
   * This method is a trigger for model component clicking.
   */
  _modelClickBinding = () => {
    // Click trigger of model <input> element
    $('input:radio[name=m]').click(() => {
      $('.alert').hide();
      $('.offload').hide();
      let um = $('input:radio[name="m"]:checked').attr('id');
      this._setModelId(um);
      const modelClasss = getModelListByClass();
      const seatModelClass = $('#' + um).parent().parent().attr('id');
      if (modelClasss.length > 1) {
        for (let modelClass of modelClasss) {
          if (seatModelClass !== modelClass) {
            let modelName = $('.model[id=' + modelClass + '] input:radio[checked="checked"]').attr('id');
            if (typeof modelName !== 'undefined') {
              um = um + '+' + modelName;
              this._setModelId(um);
            }
          }
        }
      }
      this._updateHistoryEntryURL();
      updateModelComponentsStyle(um);
      this.main();
    });
  };

  /**
   * This method is to show common UI parts.
   */
  _commonUI = () => {
    const frameworkList = getFrameworkList(this._inferenceModels);
    if (frameworkList.length > 1) {
      setFrameworkComponents(frameworkList, this._currentFramework);
    }

    this._showDynamicComponents();

    // Click trigger of framework <input> element
    $('input:radio[name=framework]').click(() => {
      $('.alert').hide();
      this._clearTimeout();
      let framework = $('input:radio[name="framework"]:checked').attr('value');
      updateFrameworkComponentsStyle(framework);
      if (framework === 'OpenCV.js') {
        $('.backend').hide();
        $('.offload').hide();
        $('.opencvjsbackend').show();
        this._setRuntimeInitialized(false);
        const opencvModel = selectModelFromGivenModels(this._inferenceModels, framework);
        if (getModelFromGivenModels(opencvModel, this._currentModelId) === null) {
          this._setModelId('none');
        }
        this._setFramework(framework);
        this._runner = null;
        this._resetOutput();
        if (typeof cv !== 'undefined') {
          this._updateHistoryEntryURL();
          this._showDynamicComponents();
          this._modelClickBinding();
          this._setRuntimeInitialized(true);
          this.main();
          return;
        }
        if (typeof this._currentOpenCVJSBackend === 'undefined') {
          this._setOpenCVJSBackend('WASM');
          showOpenCVRuntimeProgressComponent();
          asyncLoadScript(`../util/opencv.js/WASM/opencv.js`, this.onOpenCvReady);
        }
        this._updateHistoryEntryURL();
        this._showDynamicComponents();
        this._modelClickBinding();
        updateTitleComponent(this._currentOpenCVJSBackend, null, this._currentModelId, this._inferenceModels);
      } else if (framework === 'WebNN') {
        $('.opencvjsbackend').hide();
        $('.backend').show();
        $('#progressruntime').hide();
        this._setFramework(framework);
        this._setRuntimeInitialized(true);
        this._updateHistoryEntryURL();
        this._showDynamicComponents();
        this._modelClickBinding();
        this._runner = null;
        this._resetOutput();
        this.main();
      }
    });

    this._modelClickBinding();

    if (this._currentInputType === 'image' || this._currentInputType === 'audio') {
      $('.nav-pills li').removeClass('active');
      $('.nav-pills #img').addClass('active');
      $('#cameratab').removeClass('active');
      $('#imagetab').addClass('active');
      $('#fps').html('');
      $('#fps').hide();
    } else {
      $('.nav-pills li').removeClass('active');
      $('.nav-pills #cam').addClass('active');
      $('#imagetab').removeClass('active');
      $('#cameratab').addClass('active');
      $('#fps').show();
    }

    // Click trigger of Dual Backend <input> element
    $('#backendswitch').click(() => {
      $('.alert').hide();
      const polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
      const webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');
      $('.b-polyfill input').removeAttr('checked');
      $('.b-polyfill label').removeClass('checked');
      $('.b-webnn input').removeAttr('checked');
      $('.b-webnn label').removeClass('checked');
      if (!isBackendSwitch()) {
        $('.backendtitle').html('Backend');
        if (polyfillId) {
          $('#' + polyfillId).attr('checked', 'checked');
          $('#l-' + polyfillId).addClass('checked');
          this._setBackend(polyfillId);
          this._setPrefer('none');
        } else if (webnnId) {
          $('#' + webnnId).attr('checked', 'checked');
          $('#l-' + webnnId).addClass('checked');
          this._setBackend('WebML');
          this._setPrefer(webnnId);
        } else {
          $('#WASM').attr('checked', 'checked');
          $('#l-WASM').addClass('checked');
          this._setBackend('WASM');
          this._setPrefer('none');
        }
      } else {
        $('.backendtitle').html('Backends');
        if (polyfillId && webnnId) {
          $('#' + polyfillId).attr('checked', 'checked');
          $('#l-' + polyfillId).addClass('checked');
          $('#' + webnnId).attr('checked', 'checked');
          $('#l-' + webnnId).addClass('checked');
          this._setBackend(polyfillId);
          this._setPrefer(webnnId);
        } else if (polyfillId) {
          $('#' + polyfillId).attr('checked', 'checked');
          $('#l-' + polyfillId).addClass('checked');
          $('#fast').attr('checked', 'checked');
          $('#l-fast').addClass('checked');
          this._setBackend(polyfillId);
          this._setPrefer('fast');
        } else if (webnnId) {
          $('#WASM').attr('checked', 'checked');
          $('#l-WASM').addClass('checked');
          $('#' + webnnId).attr('checked', 'checked');
          $('#l-' + webnnId).addClass('checked');
          this._setBackend('WASM');
          this._setPrefer(webnnId);
        } else {
          $('#WASM').attr('checked', 'checked');
          $('#l-WASM').addClass('checked');
          $('#fast').attr('checked', 'checked');
          $('#l-fast').addClass('checked');
          this._setBackend('WASM');
          this._setPrefer('fast');
        }
      }
      updateBackendComponents(this._currentBackend, this._currentPrefer);
      this._updateHistoryEntryURL();
      this.main();
    });

    // Click trigger of Polyfill Backend <input> element
    $('input:radio[name=bp]').click(() => {
      $('.alert').hide();
      let polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
      if (isBackendSwitch()) {
        if (polyfillId !== this._currentBackend) {
          $('.b-polyfill input').removeAttr('checked');
          $('.b-polyfill label').removeClass('checked');
          $('#' + polyfillId).attr('checked', 'checked');
          $('#l-' + polyfillId).addClass('checked');
        } else if (this._currentPrefer === 'none') {
          showAlertComponent('At least one backend required, please select other backends if needed.');
          return;
        } else {
          $('.b-polyfill input').removeAttr('checked');
          $('.b-polyfill label').removeClass('checked');
          polyfillId = 'WebML';
        }
        this._setBackend(polyfillId);
      } else {
        $('.b-polyfill input').removeAttr('checked');
        $('.b-polyfill label').removeClass('checked');
        $('.b-webnn input').removeAttr('checked');
        $('.b-webnn label').removeClass('checked');
        $('#' + polyfillId).attr('checked', 'checked');
        $('#l-' + polyfillId).addClass('checked');
        this._setBackend(polyfillId);
        this._setPrefer('none');
      }
      updateBackendComponents(this._currentBackend, this._currentPrefer);
      this._updateHistoryEntryURL();
      this.main();
    });

    // Click trigger of Native Backend <input> element
    $('input:radio[name=bw]').click(() => {
      $('.alert').hide();
      let webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');
      if (isBackendSwitch()) {
        if (webnnId !== this._currentPrefer) {
          $('.b-webnn input').removeAttr('checked');
          $('.b-webnn label').removeClass('checked');
          $('#' + webnnId).attr('checked', 'checked');
          $('#l-' + webnnId).addClass('checked');
        } else if (this._currentBackend === 'WebML') {
          this.showAlertComponent('At least one backend required, please select other backends if needed.');
          return;
        } else {
          $('.b-webnn input').removeAttr('checked');
          $('.b-webnn label').removeClass('checked');
          webnnId = 'none';
        }
      } else {
        $('.b-polyfill input').removeAttr('checked');
        $('.b-polyfill label').removeClass('checked');
        $('.b-webnn input').removeAttr('checked');
        $('.b-webnn label').removeClass('checked');
        $('#' + webnnId).attr('checked', 'checked');
        $('#l-' + webnnId).addClass('checked');
        this._setBackend('WebML');
      }
      this._setPrefer(webnnId);
      updateBackendComponents(this._currentBackend, this._currentPrefer);
      this._updateHistoryEntryURL();
      this.main();
    });

    // Click trigger of opencvjsbackend <input> element
    $('input:radio[name=opencvjsbackend]').click(() => {
      $('.alert').hide();
      let selectedBackend = $('input:radio[name="opencvjsbackend"]:checked').attr('value');
      updateOpenCVJSBackendComponentsStyle(selectedBackend);
      this._setRuntimeInitialized(false);
      this._setOpenCVJSBackend(selectedBackend);
      const locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
      window.history.pushState(null, null, locSearch);
      window.location.reload(true);
      this._updateHistoryEntryURL();
    });

    // Click trigger to do inference with <img> element
    $('#img').click(() => {
      $('.alert').hide();
      $('#fps').html('');
      $('#fps').hide();
      $('ul.nav-pills li').removeClass('active');
      $('ul.nav-pills #img').addClass('active');
      $('#imagetab').addClass('active');
      $('#cameratab').removeClass('active');
      this._setInputElement(this._feedElement);
      let inputType = this._getDefaultInputType();
      this._setInputType(inputType);
      this._updateHistoryEntryURL();
      this.main();
    });

    // Click trigger to do inference with <video> or <audio> media element
    $('#cam').click(() => {
      $('.alert').hide();
      $('ul.nav-pills li').removeClass('active');
      $('ul.nav-pills #cam').addClass('active');
      $('#cameratab').addClass('active');
      $('#imagetab').removeClass('active');
      this._setInputElement(this._feedMediaElement);
      let mediaType = this._getDefaultInputMediaType();
      this._setInputType(mediaType);
      this._updateHistoryEntryURL();
      this._feedMediaElement.onloadeddata = this.main();
    });

    // Click trigger to display or hide controls components
    $('#extra').click(() => {
      componentToggle();
      this._setHiddenControlsFlag();
      this._updateHistoryEntryURL();
    });

    let inputFileElement = document.getElementById('input');
    inputFileElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        this._feedElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    $('.offload').hide();

    this._commonUIExtra();
  };

  /**
   * This method is to show custom UI parts except common UI parts.
   */
  _customUI = () => {};

  /**
   * This method is to prepare components on UI and set some click trigger bindings.
   * Compoents including:
   *  frameworks, models, backend, progress, inference result, etc. components
   * Click trigger bindings:
   *  framework click trigger bindings
   *  model element click trigger bindings
   *  backend element click trigger bindings
   *  input type element click trigger bindings
   *  some inference result displaying trigger bindings
   */
  UI = () => {
    this._updateHistoryEntryURL(location.search);

    // Show components and triggers
    this._commonUI();
    this._customUI();
  };

  /**
   * This method returns runner instance to load model/compile model/inference.
   * @returns {object} This returns a runner instance.
   */
  _createRunner = () => {
    // Override by inherited if needed
    const runner = new WebNNRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _getRunner = () => {
    // Override by inherited when example has co-work runners
    if (this._runner == null) {
      this._runner = this._createRunner();
    }
  };

  _doInitialRunner = () => {
    // Override by inherited when example has co-work runners
    const modelInfo = getModelById(this._inferenceModels.model, this._currentModelId);

    if (modelInfo != null) {
      this._setModelInfo(modelInfo);
      this._runner.doInitialization(modelInfo);
    } else {
      throw new Error('Unrecorgnized model, please check your typed url.');
    }
  };

  /** @override */
  _loadModel = async () => {
      await this._runner.loadModel(this._currentModelInfo);
  };

  /**
   * This method is to get options paramter for compiling model.
   * @returns {!Object<string, *>}
   */
  _getCompileOptions = () => {
    let options = {};

    if (this._currentFramework === 'WebNN') {
        options.backend = this._currentBackend;
        options.prefer = this._currentPrefer;
        const supportedOps = getSupportedOps(this._currentBackend, this._currentPrefer);
        options.supportedOps = supportedOps;
        options.eagerMode = false;
    }

    return options;
  };

  /** @override */
  _compileModel = async () => {
    let options = await this._getCompileOptions();
    await this._runner.compileModel(options);
  };

  /** @override */
  _predict = async () => {
    const input = {
      src: this._currentInputElement,
      options: {
        inputSize: this._currentModelInfo.inputSize,
        preOptions: this._currentModelInfo.preOptions,
        imageChannels: 4,
      },
    };
    await this._runner.run(input);
    this._postProcess();
  };

  /**
   * This method returns media stream for predicting stream.
   * @returns {!MediaStream} This returns a MediaStream.
   */
  _getMediaStream = () => {};

  /**
   * This method is to predict the frame of camera video.
   * @param {!MediaStream} stream: A MediaStream object that for stream which is used by those examples relevants with video or audio.
   */
  _predictFrame = async (stream) => {
    if (this._isStreaming)  {
      this._stats.begin();
      await this._predict();
      this._stats.end();
      let timeoutId = setTimeout(this._predictFrame, 0);
      this._setTimeoutId(timeoutId);
    }
  };

  /**
   * This method is to predict camera video or microphone audio.
   */
  _predictStream = async () => {
    let stream = await this._getMediaStream();
    this._currentInputElement.srcObject = stream;
    this._setTrack(stream.getTracks()[0]);
    await showProgressComponent('done', 'done', 'current'); // 'COMPLETED_COMPILATION'
    await this._predictFrame(stream);
    await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
    readyShowResultComponents();
  };

  /**
   * This method is to set supported ops for WebNN model compliation.
   * @param {!Array<number>} ops
   */
  _setSupportedOps = (ops) => {
    this._runner.setSupportedOps(ops);
  };

  /**
   * This method returns required ops of used model.
   * @returns {!Array<number>} This returns an array object for required ops.
   */
  _getRequiredOps = () => {
    return this._runner.getRequiredOps();
  };

  /**
   * This method returns inference subgraphs summary info of used model with hybrid backend.
   * @returns {object} This returns an array object for subgraphs summary info.
   */
  _getSubgraphsSummary = () => {
    return this._runner.getSubgraphsSummary();
  };

  /**
   * This method is doing clearing extra output on UI by example:
   * details referring to '_postProcess' method
   */
  _resetExtraOutput = () => {
    // Override by inherited if needed
  };

  /**
   * This method is to doing extra post processing by examples,
   * details referring to '_postProcess' method
   * @param output: An object for inference result and relevant info.
   */
  _processExtra = (output) => {
    // Override by inherited if needed
  };

  /**
   * This method is doing reset actions for output on UI:
   * 1. clear inference time
   * 2. clear extra output, details referring to '_postProcess' method
   */
  _resetOutput = () => {
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = '';
    this._resetExtraOutput();
  };

  /**
   * This method is doing following post processing with inference result:
   * 1. show inference time, if example used co-model to inference,
   *    the inference time is cumulative inference time by each model
   * 2. show extra output by examples, likes:
   *      image classified Top3 labels & probabilities results for image classfication example
   *      bounding boxes, classified labels & probabilities results for object detection example
   *      pose, bounding boxes results for skeleton detection example
   *      pixels by category labels, category labels results for semantic segmentation example
   *      bouding boxes, identification labels for face recognition
   *      bouding boxes, facail landmarks results for facail landmark detection example
   *      bouding boxes, emotion cateogory labels & probabilities results for  emotion analysis example
   *      super resolution result for super resolution example
   *      top3 command category labels & probabilities results for speech command example
   *      inference clycels, average time, reference translations and metrics results for speech examples
   * @override
   */
  _postProcess = () => {
    const output = this._runner.getOutput();
    let inferenceTime = 0;

    if (Object.keys(this._inferenceModels).length === 1) {
      inferenceTime = output.inferenceTime;
    } else {
      // cumulative inference time by multi runners
      inferenceTime = this._totalInferenceTime;
    }

    // show inference time
    console.log(`Inference time: ${inferenceTime} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${inferenceTime.toFixed(2)} ms</span>`;

    // show extra output info
    this._processExtra(output);
  };

  /**
   * This method is for running OpenCV.js framework, execute main method when OpenCV.js runtime was initialized.
   */
  onOpenCvReady = async () => {
    cv = await cv;
    $('#progressruntime').hide();
    this._setRuntimeInitialized(true);
    this.main();
  }

  /**
   * This method is to run inference according to selected framework, model and backend,
   * then shows the post processing of inference result on UI.
   */
  main = async () => {
    if (!this._runtimeInitialized) {
      console.log(`Runtime isn't initialized`);
      return;
    }
    this._setStreaming(false);
    // Update UI title component info
    if (this._currentFramework === 'WebNN') {
      updateTitleComponent(this._currentBackend, this._currentPrefer, this._currentModelId, this._inferenceModels);
    } else if (this._currentFramework === 'OpenCV.js') {
      updateTitleComponent(this._currentOpenCVJSBackend, null, this._currentModelId, this._inferenceModels);
    }

    if (this._currentModelId === 'none') {
      showErrorComponent('No model selected', 'Please select model to start prediction.');
      return;
    } else {
      if (this._currentModelId === 'mobilenet_v1_quant_caffe2') {
        let checkFlag = false;
        if (this._currentBackend !== 'WebML') {
          checkFlag = true;
        } else {
          if (this._currentPrefer !== 'fast') {
            checkFlag = true;
          }
        }

        if (checkFlag) {
          showErrorComponent('Incorrect backend', `Current model just support 'FAST_SINGLE_ANSWER' backend.`);
          return;
        }
      }

      const modelCategoryLen = Object.keys(this._inferenceModels).length;
      if (modelCategoryLen > 1) {
        if (this._currentModelId.includes('+') || this._currentModelId.includes(' ')) {
          let modelIdArray;
          if (this._currentModelId.includes('+')) {
            modelIdArray = this._currentModelId.split('+');
          } else if (this._currentModelId.includes(' ')) {
            modelIdArray = this._currentModelId.split(' ');
          }
          if (modelCategoryLen > modelIdArray.length) {
            showErrorComponent('Not enough selected models', `Please select ${modelCategoryLen} kinds of models to start prediction.`);
            return;
          }
        } else {
          showErrorComponent('Not enough selected models', `Please select ${modelCategoryLen} kinds of models to start prediction.`);
          return;
        }
      }
    }
    try {
      // Get Runner for execute inference
      this._getRunner();
      // intial runner
      this._doInitialRunner();
      // UI shows loading model progress
      await showProgressComponent('current', 'pending', 'pending');
      // Load model
      await this._loadModel();
      // UI shows compiling model progress, includes warm up model
      await showProgressComponent('done', 'current', 'pending');
      // Compile model
      await this._compileModel();
      if (this._currentFramework === 'WebNN') {
        const requiredOps = this._getRequiredOps();
        // show offload ops info
        const supportedOps = getSupportedOps(this._currentBackend, this._currentPrefer);
        showHybridComponent(supportedOps, requiredOps, this._currentBackend, this._currentPrefer);
        // show sub graphs summary
        const subgraphsSummary = this._getSubgraphsSummary();
        showSubGraphsSummary(subgraphsSummary);
      }
      // UI shows inferencing progress
      await showProgressComponent('done', 'done', 'current');
      // Inference with Web NN API
      switch (this._currentInputType) {
        case 'image':
        case 'audio':
          // Stop webcam opened by navigator.getUserMedia
          if (this._track != null) {
            this._track.stop();
          }
          await this._predict();
          await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
          readyShowResultComponents();
          break;
        case 'camera':
        case 'microphone':
          this._setStreaming(true);
          await this._predictStream();
          break;
        default:
        // Never goes here
      }
    } catch (e) {
      showAlertComponent(e);
      showErrorComponent();
    }
  };
};
