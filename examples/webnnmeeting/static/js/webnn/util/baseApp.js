class baseApp {
  constructor(name, modelLists, front) {
    this.appName = name;
    this.titleName = name; // ? sync with Belem
    this.modelLists = modelLists;
    this.front = front;

    this.currentBackend;
    this.currentModelId;
    this.currentPrefer;
    this.currentDisplay;
    this.feedSrcType;

    this.modelInfo = null;
    this.runner = null;
    this.streaming = false;
    this.stats = new Stats();
    this.track;
    this.modelString = null;
  
    this.progressBar = document.getElementById('progressBar');
  }

  // UI Parts
  setBackend = (backend) => {
    this.currentBackend = backend;
  }

  setPrefer = (prefer) => {
    this.currentPrefer = prefer;
  }

  setModelString = (modelString) => {
    this.modelString = modelString;
  }

  setTitleName = (name) => {
    this.titleName = name;
  }

  setFeedSrcType = (src) => {
    this.feedSrcType = src;
  }

  setDisplay = (display) => {
    if (display !== undefined) {
      this.currentDisplay = display;
    } else if (this.currentDisplay == '1') {
      this.currentDisplay = '0';
    } else {
      this.currentDisplay = '1';
    }
  }

  setFront = () =>  {
    this.front = !this.front;
  }

  parseSearchParams = () => {
    let up = getUrlParam('prefer');
    this.setPrefer(up);
  
    let ub = getUrlParam('b');
    this.setBackend(ub);
  
    let um = getUrlParam('m');
    this.setModelString(um);
  
    let us = getUrlParam('s');
    this.setFeedSrcType(us);
  
    let ud = getUrlParam('d');
    this.setDisplay(ud);
  } 
 
  updateLoadingComponent = (loadedSize, totalSize, percentComplete) => {
    $('.loading-page .counter h1').html(`${loadedSize}/${totalSize}MB ${percentComplete}%`);
  }

  updateScenario = async (camera = false) => {
    this.streaming = false;
    this.predictByRunner();
  }

  updateBackend = async (force = false) => {
    if (force) {
      this.runner.initialized = false;
    }
    this.streaming = false;
    try { 
      this.runner.deleteAll(); 
    } catch (e) { 

    }
    await this.showProgressComponent('done', 'current', 'pending');
    try {
      this.showHybridComponent();
      await this.initRunner();
      showSubGraphsSummary(this.runner.getSubgraphsSummary());
      this.predictByRunner();
    }
    catch (e) {
      this.errorHandler(e);
    }
  }

  updateNativeBackend = () => {
    if(this.currentPrefer === 'sustained' & currentOS === 'Mac OS') {
      let nativebackend = getNativeAPI(this.currentPrefer);
      $('#l-sustained').html('SUSTAINED_SPEED');
      $('#l-sustained').append(`<span class='nbackend'>${nativebackend}</span>`);
    } else {
      $('#l-sustained').html('SUSTAINED_SPEED');
    }
  }

  updateTitleComponent = () => {
    let currentprefertext = {
      fast: 'FAST',
      sustained: 'SUSTAINED',
      low: 'LOW',
      none: 'None',
    }[this.currentPrefer];
    let backendtext = this.currentBackend;
    if (backendtext == 'WebML') {
      backendtext = 'WebNN';
    }
    if (this.currentBackend !== 'WebML' && this.currentPrefer !== 'none') {
      backendtext = this.currentBackend + ' + WebNN';
    }
    let modelShow = null;
    let modelIdArray;
    if (this.modelString.includes('+') || this.modelString.includes(' ')) {
      if (this.modelString.includes('+')) {
        modelIdArray = this.modelString.split('+');
      } else if (this.modelString.includes(' ')) {
        modelIdArray = this.modelString.split(' ');
      }
      for (let model of modelIdArray) {
        if (modelShow === null) {
          modelShow = getModelById(model).modelName;
        } else {
          modelShow = modelShow + ' + ' + getModelById(model).modelName;
        }
      }
    } else {
      modelShow = getModelById(this.modelString).modelName;
    }
    if(currentprefertext === 'None') {
      $('#ictitle').html(`${this.appName} / ${backendtext} / ${modelShow || 'None'}`);
    } else if(this.currentPrefer === 'sustained' & currentOS === 'Mac OS') {
      $('#ictitle').html(`${this.appName} / ${backendtext} (${currentprefertext}/MPS) / ${modelShow || 'None'}`);
    } else {
      $('#ictitle').html(`${this.appName} / ${backendtext} (${currentprefertext}) / ${modelShow || 'None'}`);
    }
  }

  changeModel = () => {
    $('.alert').hide();
    let um = $('input:radio[name="m"]:checked').attr('id');
    if (this.modelString === um) {
      return;
    }
    this.setModelString(um);
    let modelClasss = getModelClasss();
    let seatModelClass = $('#' + um).parent().parent().attr('id');
    if (modelClasss.length > 1) {
      for (let modelClass of modelClasss) {
        if (seatModelClass !== modelClass) {
          let modelName = $('.model[id=' + modelClass + '] input:radio[checked="checked"]').attr('id');
          if (typeof modelName !== 'undefined') {
            um = um + '+' + modelName;
          }
        }
      }
      this.setModelString(um);
    }
    strsearch = `?prefer=${this.currentPrefer}&b=${this.currentBackend}&m=${this.modelString}&s=${this.feedSrcType}&d=${this.currentDisplay}`;
    window.history.pushState(null, null, strsearch);
    this.checkedModelsComponent();
    this.disabledModelsComponent();
    this.updateTitleComponent();
    $('.offload').hide();
    if (modelClasss.length > 1) {
      let umArray = um.split('+');
      if (modelClasss.length === umArray.length) {
        this.main();
      } else {
        this.showError('Not enough selected models', 'Please select ' + modelClasss.length + ' kinds of models to start prediction.');
      }
    } else {
      this.main();
    }
  };

  showHybridComponent = async () => {
    const hybridRow = (offloadops, backend, prefer) => {
      if(offloadops && offloadops.size > 0 && backend != 'WebML' && prefer != 'none') {
        $('.offload').fadeIn();
        let offloadopsvalue = '';
        offloadops.forEach((value) => {
          let t = '<span class="ol">' + operationTypes[value] + '</span>';
          offloadopsvalue += t;
        })
        $(".ol").remove();
        $("#offloadops").html(`Following ops were offloaded to <span id='nnbackend' class='ols'></span> from <span id='polyfillbackend' class='ols'></span>: `);
        $("#offloadops").append(offloadopsvalue).append(`<span data-toggle="modal" class="subgraph-btn" data-target="#subgraphModal">View Subgraphs</span>`);
        $("#nnbackend").html(prefer);
        $("#polyfillbackend").html(backend);
      } else {
        $('.offload').hide();
      }
    };

    // update the global variable `supportedOps` defined in the base.js
    supportedOps = getWebNNSupportedOps(this.currentBackend, this.currentPrefer);
    let requiredOps = await this.getRequiredOps();
    let intersection = new Set([...supportedOps].filter(x => requiredOps.has(x)));
    console.log('NN supported: ' + [...supportedOps]);
    console.log('Model required: ' + [...requiredOps]);
    console.log('Ops offload: ' + [...intersection]);
    hybridRow(intersection, this.currentBackend, this.currentPrefer);
  }

  showAlert = (error) => {
    console.error(error);
    let div = document.createElement('div');
    div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>${error}</strong>`;
    div.innerHTML += `<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  updateProgress = (ev) => {
    if (ev.lengthComputable) {
      let totalSize = ev.total / (1000 * 1000);
      let loadedSize = ev.loaded / (1000 * 1000);
      let percentComplete = ev.loaded / ev.total * 100;
      percentComplete = percentComplete.toFixed(0);
      this.progressBar.style = `width: ${percentComplete}%`;
      this.updateLoadingComponent(loadedSize.toFixed(1), totalSize.toFixed(1), percentComplete);
    }
  }

  errorHandler = (e) => {
    this.showAlert(e);
    this.showError(null, null);
  }

  runPredict = async (source) => {
    let inputSize = this.modelInfo.inputSize;
    let options = {
      inputSize: this.modelInfo.inputSize,
      preOptions: this.modelInfo.preOptions || {},
      imageChannels: 4, // RGBA
      drawWH: [inputSize[1], inputSize[0]],
    };
    let ret = await this.runner.predict(source, options);
    return ret;
  }

  startPredictCamera = async () => {
    if (this.streaming) {
      try {
        this.stats.begin();
        let ret = await this.runPredict(this.liveSrcElement);     
        this.handleInferencedResult(ret, this.liveSrcElement);
        this.stats.end();
        setTimeout(this.startPredictCamera, 0);
      } catch (e) {
        this.errorHandler(e);
      }
    }
  }

  predictByRunner = async () => {
    try {
      if (this.feedSrcType == 'camera') {
        this.streaming = true;
        await this.showProgressComponent('done', 'done', 'current');
        let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: (this.front ? 'user' : 'environment') } });
        video.srcObject = stream;
        this.track = stream.getTracks()[0];
        this.startPredictCamera();
        await this.showProgressComponent('done', 'done', 'done');
        this.showResults();
      } else {
        this.streaming = false;
        // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
        if (this.track) {
          this.track.stop();
        }
        await this.showProgressComponent('done', 'done', 'current');
        let ret = await this.runPredict(this.srcElement);
        await this.showProgressComponent('done', 'done', 'done');
        this.showResults();
        this.handleInferencedResult(ret, this.srcElement);
      }
    } catch (e) {
      this.errorHandler(e);
    }
  }

  handleInferencedResult = (result, source) => {
    const showInferenceTime = (time) => {
      try {
        let inferenceTime = time.toFixed(2);
        console.log(`Inference time: ${inferenceTime} ms`);
        let inferenceTimeElement = document.getElementById('inferenceTime');
        inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${inferenceTime} ms</span>`;
      } catch (e) {
        console.log(e);
      }
    }
  
    try {
      showInferenceTime(result.time);
      this.drawResultComponents(result.drawData, source);
    } catch (e) {
      console.log(e);
    }
  }

  initRunner = async () => {
    // Be implemented when App have more runners
    let init = await this.runner.initModel(this.currentBackend, this.currentPrefer);
    if (init == 'NOT_LOADED') {
      return;
    }
  }

  getRequiredOps = async () => {
    let requiredOps = await this.runner.getRequiredOps();
    return requiredOps;
  }

  loadModel = async () => {
    await this.runner.loadModel();
  }

  createRunner = () => {
    // To be implemented by each App
  }

  main = async () => {
    if (this.currentDisplay != '0') {
      componentToggle();
    }
    this.disabledModelsComponent();
    if (this.modelString === 'none') {
      this.showError('No model selected', 'Please select a model to start prediction.');
      return;
    }
    await this.showProgressComponent('current', 'pending', 'pending');
    try {
      if (this.createRunner() === 'SUCCESS') {
        await this.loadModel();
        this.showHybridComponent();
        await this.showProgressComponent('done', 'current', 'pending');
        await this.initRunner();
        showSubGraphsSummary(this.runner.getSubgraphsSummary());
      }
    } catch (e) {
      this.errorHandler(e);
    }
    this.predictByRunner();
  }
}
