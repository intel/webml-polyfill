class SpeechCommandsExample extends BaseMircophoneExample {
  constructor(models) {
    super(models);
  }

  /** @override */
  _customUI = () => {
    let _this = this;
    let inputFileElement = document.getElementById('input');
    inputFileElement.addEventListener('change', (e) => {
      $('#controller div').removeClass('current');
      this.main();
    }, false);

    $('#controller div').click(function () {
      let t = $(this).attr('id');
      _this._currentInputElement.src = `audio/${t}.wav`;
      _this._currentInputElement.play();
      _this.main();
      $('#controller div').removeClass('current');
      $(`#${t}`).addClass('current');
    });

    const recordButton = document.getElementById('record');
    recordButton.addEventListener('click', () => {
      _this._predictStream();
    }, false);

    Module.onRuntimeInitialized = () => {
      console.log('WASM runtime is ready for mfccs.');
    };
  };

  /** @override */
  _createRunner = () => {
    let runner;
    switch (this._currentFramework) {
      case 'WebNN':
        runner = new WebNNRunner();
        break;
      case 'OpenVINO.js':
        runner = new OpenVINORunner();
        break;
    }
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _predict = async () => {
    try {
      this._stats.begin();
      const input = {
        src: this._currentInputElement,
        options: {
          inputSize: this._currentModelInfo.inputSize,
          sampleRate: this._currentModelInfo.sampleRate,
          mfccsOptions: this._currentModelInfo.mfccsOptions,
        },
      };
      await this._runner.run(input);
      this._postProcess();
      this._stats.end();
    } catch (e) {
      showAlertComponent(e);
      showErrorComponent();
    }
  }

  _handleDataAvailable = async (e) => {
    let buffer = [];
    buffer.push(e.data);
    let blob = new Blob(buffer, { type: 'audio/wav' });
    this._feedMediaElement.src = window.URL.createObjectURL(blob);
    this._setInputElement(this._feedMediaElement);
    await this._predict();
    await showProgressComponent('done', 'done', 'done'); // 'COMPLETED_INFERENCE'
    readyShowResultComponents();
    this._feedMediaElement.play();
  }

  _recordAndPredictMicrophone = async (stream) => {
    let audioRecorder = new MediaRecorder(stream, { audio: true });
    audioRecorder.ondataavailable = this._handleDataAvailable;
    audioRecorder.start();
    await new Promise(() => setTimeout(() => {
      audioRecorder.stop();
      // Stop webmic opened by navigator.getUserMedia after record
      if (this._track) {
        this._track.stop();
      }
    }, 1000));
  };

  /** @override */
  _predictStream = async () => {
    const constraints = this._getMediaConstraints();
    let stream = await navigator.mediaDevices.getUserMedia(constraints);
    this._setTrack(stream.getTracks()[0]);
    await new Promise(() => setTimeout(this._recordAndPredictMicrophone(stream), 500));
  };

  /** @override */
  _processExtra = (output) => {
    const deQuantizeParams = this._runner.getDeQuantizeParams();
    const outputTensor = postOutputTensorAudio(output.tensor);
    const labelClasses = getTopClasses(outputTensor, output.labels, 3, deQuantizeParams);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });

    if (labelClasses[0].prob > 50) {
      $('#speechcommands #scresult svg').removeClass('current');
      $(`#r${labelClasses[0].label}`).addClass('current');
      $('#speechcommands #scresult #rtext').html(`${labelClasses[0].label}`);
    } else {
      $('#speechcommands #scresult svg').removeClass('current');
      $('#speechcommands #scresult #rtext').html(`Unknown`);
    }
  };
}