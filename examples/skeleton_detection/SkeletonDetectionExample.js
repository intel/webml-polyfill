class SkeletonDetectionExample {
  constructor(models) {
    this._currentModelInfo = models.model[0];
    this._currentInputType = 'image'; // input type: image, camera
    // <image> or <audio> element
    this._feedElement = document.getElementById('feedElement');
    // <video> or <audio> element when using live Camera or Microphone
    this._feedMediaElement = document.getElementById('feedMediaElement');
    // track and stats serve for 'VIDEO' | 'AUDIO' input
    this._track = null;
    this._stats = new Stats();
    // _hiddenControlsFlag ('0'/'1') is for UI shows/hides model & backend control
    this._hiddenControlsFlag = '0';
    this._currentInputElement = null;
    // Backend type: 'WASM' | 'WebGL' | 'WebML'
    this._currentBackend = 'WASM';
    // Prefer type: 'none' | 'fast' | 'sustained' | 'low'
    this._currentPrefer = 'none';
    this._modelConfig = {
      version: 1.01,
      outputStride: 16,
      scaleFactor: 0.5,
      useAtrousConv: true,
    };
    this._showConfig = {
      minScore: 0.15,
      maxDetections: 15,
      nmsRadius: 20.0,
      showPose: true,
      showBoundingBox: false,
    };
    this._bFrontCamera = false;
    this._runner = null;
    this._isStreaming = false; // for inference camera video
  }

  _setInputType = (t) => {
    this._currentInputType = t;
  };

  _setInputElement = (element) => {
    this._currentInputElement = element;
  };

  useFrontFacingCamera = (flag) => {
    if (typeof flag == "undefined") {
      this._bFrontCamera = !this._bFrontCamera;
    } else {
      this._bFrontCamera = flag;
    }
  };

  _setBackend = (backend) => {
    this._currentBackend = backend;
  };

  _setPrefer = (prefer) => {
    this._currentPrefer = prefer;
  };

  _setTrack = (track) => {
    this._track = track;
  };

  _setStreaming = (flag) => {
    this._isStreaming = flag;
  };

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

  _updateHistoryEntryURL = (url) => {
    if (typeof url !== 'undefined') {
      if (url === "") {
        this._setBackend('WASM');
        this._setPrefer('none');
        this._setInputType('image');
        this._setInputElement(this._feedElement);
        this._setHiddenControlsFlag('0');
      } else {
        // Parse seach params, and prepare inference env
        const prefer = parseSearchParams('prefer');
        this._setPrefer(prefer);
        const backend = parseSearchParams('b');
        this._setBackend(backend);
        let feedEle = null;
        const inputType = parseSearchParams('s');
        this._setInputType(inputType);
        switch (inputType.toLowerCase()) {
          case 'image':
            feedEle = this._feedElement;
            break;
          case 'camera':
            feedEle = this._feedMediaElement;
            break;
          default:
            showErrorComponent(`Invalid url`, "Invalide value of 's' parameter of typed url.It requires 'file' or 'stream' .");
            return;
        }
        this._setInputElement(feedEle);
        const flag = parseSearchParams('d');
        this._setHiddenControlsFlag(flag);
      }
    }

    const locSearch = `?prefer=${this._currentPrefer}&b=${this._currentBackend}&s=${this._currentInputType}&d=${this._hiddenControlsFlag}`;
    window.history.pushState(null, null, locSearch);
  };

  UI = () => {
    this._updateHistoryEntryURL(location.search);

    //set backend components
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

    if (this._currentInputType === 'image') {
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
      this._setInputType('image');
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
      this._setInputType('camera');
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

    if (/Mobi|Android|iPhone|iPad|iPod/.test(navigator.userAgent)) {
      // for mobile devices: smartphone, pad
      if (this._currentInputType === 'camera') {
        $('#cameraswitcher').show();
      } else {
        $('#cameraswitcher').hide();
      }

      $('#cameraswitch').prop('checked', this._bFrontCamera);

      $('#img').click(() => {
        $('#cameraswitcher').hide();
        this._setInputElement(this._feedElement);
      });

      $('#cam').click(() => {
        $('#cameraswitcher').fadeIn();
        this._setInputElement(this._feedMediaElement);
      });

      $('#cameraswitch').click(() => {
        $('.alert').hide();
        this.useFrontFacingCamera();
        $('#cameraswitch').prop('checked', this._bFrontCamera);
        this.main();
      });

      $('#fullscreen i svg').click(() => {
        $('#cameraswitcher').toggleClass('fullscreen');
      })
    } else {
      $('#cameraswitcher').hide();
    }

    $('#cam').click(() => {
      $('#fps').show();
    });

    $('#fullscreen i svg').click(() => {
      const toggleFullScreen = () => {
        let doc = window.document;
        let docEl = doc.documentElement;
        let requestFullScreen = docEl.requestFullscreen || docEl.mozRequestFullScreen || docEl.webkitRequestFullScreen || docEl.msRequestFullscreen;
        let cancelFullScreen = doc.exitFullscreen || doc.mozCancelFullScreen || doc.webkitExitFullscreen || doc.msExitFullscreen;
        if (!doc.fullscreenElement && !doc.mozFullScreenElement && !doc.webkitFullscreenElement && !doc.msFullscreenElement) {
          requestFullScreen.call(docEl);
        } else {
          cancelFullScreen.call(doc);
        }
      };

      $('#fullscreen i').toggle();
      toggleFullScreen();
      $('#feedMediaElement').toggleClass('fullscreen');
      // $('#overlay').toggleClass('video-overlay');
      $('#fps').toggleClass('fullscreen');
      $('#fullscreen i').toggleClass('fullscreen');
      $('#ictitle').toggleClass('fullscreen');
      $('#inference').toggleClass('fullscreen');
      $('#canvasvideo').toggleClass('fullscreen');
      $('#my-gui-container').toggleClass('fullscreen');
    });

    this._feedElement.addEventListener('load', () => {
      this.main();
    }, false);

    $('#option').show();

    $('#sdmodel').change(() => {
      this._modelConfig.version = $('#sdmodel').find('option:selected').attr('value');
      this.main();
    });

    $('#sdstride').change(() => {
      this._modelConfig.outputStride = parseInt($('#sdstride').find('option:selected').attr('value'));
      this.main();
    });

    $('#scalefactor').change(() => {
      this._modelConfig.scaleFactor = parseFloat($('#scalefactor').find('option:selected').attr('value'));
      this.main();
    });

    $('#sdscorethreshold').change(() => {
      this._showConfig.minScore = parseFloat($('#sdscorethreshold').val());
      this._postProcess();
    });

    $('#sdnmsradius').change(() => {
      this._showConfig.nmsRadius = parseInt($('#sdnmsradius').val());
      this._postProcess();
    });

    $('#sdmaxdetections').change(() => {
      this._showConfig.maxDetections = parseInt($('#sdmaxdetections').val());
      this._postProcess();
    });

    $('#sdshowpose').change(() => {
      this._showConfig.showPose = $('#sdshowpose').prop('checked');
      this._postProcess();
    });

    $('#sduseatrousconvops').change(() => {
      this._modelConfig.useAtrousConv = $('#sduseatrousconvops').prop('checked');
      this.main();
    });

    $('#sdshowboundingbox').change(() => {
      this._showConfig.showBoundingBox = $('#sdshowboundingbox').prop('checked');
      this._postProcess();
    });
  };

  _predict = async () => {
    const inputSize = this._currentModelInfo.inputSize;
    const outputStride = Number(this._modelConfig.outputStride);
    const scaleFactor = this._modelConfig.scaleFactor;
    const scaleWidth = getValidResolution(scaleFactor, inputSize[1], outputStride);
    const scaleHeight = getValidResolution(scaleFactor, inputSize[0], outputStride);
    const input = {
      src: this._currentInputElement,
      options: {
        inputSize: [scaleHeight, scaleWidth, inputSize[2]],
        preOptions: this._currentModelInfo.preOptions,
        imageChannels: 4,
      },
    };
    await this._runner.run(input);
    this._postProcess();
  };

  _predictFrame = async () => {
    if (this._isStreaming) {
      this._stats.begin();
      await this._predict();
      this._stats.end();
      setTimeout(this._predictFrame, 0);
    }
  };

  _predictStream = async () => {
    let stream = await navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: (this._bFrontCamera ? 'user' : 'environment')}});
    this._currentInputElement.srcObject = stream;
    this._setTrack(stream.getTracks()[0]);
    await showProgressComponent('done', 'done', 'current'); // 'COMPLETED_COMPILATION'
    await this._predictFrame();
    await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
    readyShowResultComponents();
  };

  _postProcess = () => {
    const drawPoses = (src, canvas, poses, options) => {
      const ctx = canvas.getContext('2d');
      const width = src.naturalWidth || src.videoWidth;
      const height = src.naturalHeight || src.videoHeight;
      canvas.setAttribute('width', width);
      canvas.setAttribute('height', height);
      if (src.videoWidth) {
        ctx.scale(-1, 1);
        ctx.translate(-width, 0);
      }
      ctx.drawImage(src, 0, 0, width, height);
      if (poses.length == 0) {
        return;
      }
      poses.forEach((pose) => {
        const scaleX = canvas.width / options.scaleWidth;
        const scaleY = canvas.height / options.scaleHeight;
        if (pose.score >= options.minScore) {
          if (options.showPose) {
            drawKeypoints(pose.keypoints, options.minScore, ctx, scaleX, scaleY);
            drawSkeleton(pose.keypoints, options.minScore, ctx, scaleX, scaleY);
          }
          if (options.showBoundingBox) {
            drawBoundingBox(pose.keypoints, ctx, scaleX, scaleY);
          }
        }
      });
    };

    const output = this._runner.getOutput();
    const inferenceTime = output.inferenceTime;
    console.log(`Inference time: ${inferenceTime} ms`);
    const inferenceTimeElement = document.getElementById('inferenceTime');
    const inputSize = this._currentModelInfo.inputSize;
    const outputStride = Number(this._modelConfig.outputStride);
    const scaleFactor = this._modelConfig.scaleFactor;
    const scaleWidth = getValidResolution(scaleFactor, inputSize[1], outputStride);
    const scaleHeight = getValidResolution(scaleFactor, inputSize[0], outputStride);
    const scaleInputSize = [1, scaleHeight, scaleWidth, this._currentModelInfo.inputSize[2]];
    if (this._currentInputType === 'image') {
        const singlePose = decodeSinglepose(sigmoid(output.heatmapTensor),
                                            output.offsetTensor,
                                            toHeatmapsize(scaleInputSize, outputStride),
                                            outputStride);
        const start = performance.now();
        const multiPoses = decodeMultiPose(sigmoid(output.heatmapTensor), output.offsetTensor,
                                           output.displacementFwd, output.displacementBwd,
                                           outputStride, this._showConfig.maxDetections, this._showConfig.minScore,
                                           this._showConfig.nmsRadius, toHeatmapsize(scaleInputSize, outputStride));
        const decodeTime = performance.now() - start;
        console.log(`Decode time: ${decodeTime} ms`);
        inferenceTimeElement.innerHTML = `Inference Time: <span class='ir'>${inferenceTime.toFixed(2)} ms</span> Decoding Time: <span class='ir'>${decodeTime.toFixed(2)} ms</span>`;
        let options = {
          inputHeight: this._currentModelInfo.inputSize[0],
          inputWidth: this._currentModelInfo.inputSize[1],
          scaleHeight: scaleHeight,
          scaleWidth: scaleWidth,
          minScore: this._showConfig.minScore,
          showPose: this._showConfig.showPose,
          showBoundingBox: this._showConfig.showBoundingBox,
        }
        const canvasSingle = document.getElementById('canvas');
        drawPoses(this._currentInputElement, canvasSingle, singlePose, options);
        const canvasMulti = document.getElementById('canvasMulti');
        drawPoses(this._currentInputElement, canvasMulti, multiPoses, options);
    } else {
      const start = performance.now();
      const multiPoses = decodeMultiPose(sigmoid(output.heatmapTensor), output.offsetTensor,
                                         output.displacementFwd, output.displacementBwd,
                                         outputStride, this._showConfig.maxDetections, this._showConfig.minScore,
                                         this._showConfig.nmsRadius, toHeatmapsize(scaleInputSize, outputStride));
      const decodeTime = performance.now() - start;
      console.log(`Decode time: ${decodeTime} ms`);
      inferenceTimeElement.innerHTML = `Inference Time: <span class='ir'>${inferenceTime.toFixed(2)} ms</span>`;
      let options = {
        inputHeight: this._currentModelInfo.inputSize[0],
        inputWidth: this._currentModelInfo.inputSize[1],
        scaleHeight: scaleHeight,
        scaleWidth: scaleWidth,
        minScore: this._showConfig.minScore,
        showPose: this._showConfig.showPose,
        showBoundingBox: this._showConfig.showBoundingBox,
      }
      const canvas = document.getElementById('canvasvideo');
      drawPoses(this._currentInputElement, canvas, multiPoses, options);
    }
  };

  main = async () => {
    // Update UI title component info
    updateTitleComponent(this._currentBackend, this._currentPrefer);
    this._setStreaming(false);
    try {
      if (this._runner == null) {
        this._runner = new SkeletonDetectionRunner();
      }
      // UI shows model-loading progress
      await showProgressComponent('current', 'pending', 'pending');
      await this._runner.loadAndCompileModel(this._currentBackend, this._currentPrefer, this._currentModelInfo, this._modelConfig);
      const supportedOps = getSupportedOps(this._currentBackend, this._currentPrefer);
      const requiredOps = this._runner.getRequiredOps();
      // show offload ops info
      showHybridComponent(supportedOps, requiredOps, this._currentBackend, this._currentPrefer);
      // show sub graphs summary
      const subgraphsSummary = this._runner.getSubgraphsSummary();
      showSubGraphsSummary(subgraphsSummary);
      // UI shows inferencing progress
      await showProgressComponent('done', 'done', 'current');
      // Inference with Web NN API
      switch (this._currentInputType) {
        case 'image':
          // Stop webcam opened by navigator.getUserMedia
          if (this._track != null) {
            this._track.stop();
          }
          await this._predict();
          await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
          readyShowResultComponents();
          break;
        case 'camera':
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