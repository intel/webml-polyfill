class ImageClassificationExample extends BaseExample {
  constructor(models) {
    super(models);
  }

  _getMediaStream = () => {
    let inputVideoElement = document.getElementById('feedMediaElement');
    return inputVideoElement.captureStream();
  };

  _predictStream = async () => {
    let stream = await this._getMediaStream();
    this._setTrack(stream.getTracks()[0]);
    await showProgressComponent('done', 'done', 'current'); // 'COMPLETED_COMPILATION'
    await this._predictFrame(stream);
    await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
    readyShowResultComponents();
  };

  onOpenCvReady = () => {
    cv.onRuntimeInitialized = () => {
      $('#progressruntime').hide();
      this._setRuntimeInitialized(true);
      if (this._currentInputType === 'image') {
        this.main();
      }
    }
  }

  _updateHistoryEntryURL = (url) => {
    let locSearch;
    if (typeof url !== 'undefined') {
      if (url === "") {
        asyncLoadScript(`../util/opencv.js/WASM/opencv.js`, this.onOpenCvReady);
        this._setOpenCVJSBackend('WASM');
        this._setModelId('none');
        this._setInputType('camera');
        this._setInputElement(this._feedElement);
        this._setHiddenControlsFlag('0');
        this._setFramework('OpenCV.js');
        this._setRuntimeInitialized(false);
        locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
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

        const opencvBackend = parseSearchParams('b');
        if (opencvBackend != this._currentOpenCVJSBackend) {
          showOpenCVRuntimeProgressComponent();
          asyncLoadScript(`../util/opencv.js/${opencvBackend.replace(' ', '+')}/opencv.js`, this.onOpenCvReady);
          this._setOpenCVJSBackend(opencvBackend);
        }
        locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
      }
    } else {
      locSearch = `?b=${this._currentOpenCVJSBackend}&m=${this._currentModelId}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}&f=${this._currentFramework}`;
    }
    window.history.pushState(null, null, locSearch);
  };

  UI = () => {
    $('#cameraswitcher').hide();
    this._updateHistoryEntryURL(location.search);
    this._showDynamicComponents();

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
      updateModelComponentsStyle(um);
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

    let inputVideoElement = document.getElementById('inputvideo');
    inputVideoElement.addEventListener('change', (e) => {
      this._setStreaming(false);
      $('#inference').hide();
      let files = e.target.files;
      if (files.length > 0) {
        this._feedMediaElement.src = URL.createObjectURL(files[0]);
      }
    }, false);


    this._feedMediaElement.addEventListener('play', () => {
      this.main();
    });

    this._feedMediaElement.addEventListener('pause', () => {
      this._setStreaming(false);
      // $('#inference').hide();
    });

    let inputFileElement = document.getElementById('input');
    inputFileElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        this._feedElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    if (this._currentInputType == 'image') {
      $('#fps').hide();
      this._currentInputElement.addEventListener('load', () => {
        // this.main();
        this._predict();
      }, false);
    }

  };

  /** @override */
  _createRunner = () => {
    let runner;
    switch (this._currentFramework) {
      case 'WebNN':
        runner = new WebNNRunner();
        break;
      case 'OpenCV.js':
        runner = new ImageClassificationOpenCVRunner();
        break;
    }
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _processExtra = (output) => {
    let labelClasses;
    switch (this._currentFramework) {
      case 'WebNN':
        const deQuantizeParams =  this._runner.getDeQuantizeParams();
        labelClasses = getTopClasses(output.tensor, output.labels, 3, deQuantizeParams);
        break;
      case 'OpenCV.js':
        labelClasses = getTopClasses(output.tensor, output.labels, 3);
        break;
    }
    $('#inferenceresult').show();
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };

  _resetExtraOutput = () => {
    $('#inferenceresult').hide();
    for (let i = 0; i < 3; i++) {
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = '';
      probElement.innerHTML = '';
    }
  };

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

      if(parseSearchParams('s') === 'image') {
        var time = 0
        $("#gallery .gallery-item").each(function() {
          var $this = $(this)
          setTimeout(function() {
            $("#gallery .gallery-item").removeClass('hl')
            $this.addClass('hl')
            let src = $this.children('img').attr('src')
            $('#feedElement').attr('src', src)
          }, time);
          time += 4000
        });
      }
      
    } catch (e) {
      showAlertComponent(e);
      showErrorComponent();
    }
  };

}

