async function main(camera) {
  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  const canvasElement1 = document.getElementById('canvas1');
  const canvasShowElement = document.getElementById('canvasShow');
  const faceDetectModel = document.getElementById('faceDetectModel');
  const emotionModel = document.getElementById('emotionModel');
  const selectPrefer = document.getElementById('selectPrefer');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const progressLabel = document.getElementById('progressLabel');

  let currentBackend = '';
  let currentDetectionModel;
  let currentEmotionModel;
  let currentPrefer = '';
  let streaming = false;

  let emotionAnalysis = new Utils(canvasElement1);
  let faceDetector = new FaceDetecor(canvasElement);
  faceDetector.updateProgress = updateProgress;
  emotionAnalysis.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

  function checkPreferParam() {
    if (currentOS === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend, models) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Currently ${backend} backend doesn't support ${models[0]} or ${models[1]} Model.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function showPreferAlert() {
    let div = document.createElement('div');
    div.setAttribute('id', 'preferAlert');
    div.setAttribute('class', 'alert alert-danger alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Invalid prefer, prefer should be 'fast' or 'sustained'.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function removeAlertElement() {
    let backendAlertElem =  document.getElementById('backendAlert');
    if (backendAlertElem !== null) {
      backendAlertElem.remove();
    }
    let preferAlertElem =  document.getElementById('preferAlert');
    if (preferAlertElem !== null) {
      preferAlertElem.remove();
    }
  }

  function updateBackend() {
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI(currentPrefer) : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend) {
    if (currentBackend === newBackend) {
      return;
    }
    streaming = false;
    if (newBackend !== "WebML") {
      selectPrefer.style.display = 'none';
    } else {
      selectPrefer.style.display = 'inline';
    }
    emotionAnalysis.deleteAll();
    faceDetector.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(async function() {
      try {
        await faceDetector.init(newBackend, currentPrefer);
        await emotionAnalysis.init(newBackend, currentPrefer);
        currentBackend = newBackend;
        updateBackend();
        updateModel();
        updatePrefer();
        if (!camera) {
          Predict(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }
      catch(e) {
        console.warn(`Failed to init ${emotionAnalysis.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(emotionAnalysis.model._backend, [currentDetectionModel.modelName, currentEmotionModel.modelName]);
        changeBackend('WASM');
        updatePrefer();
        backend.innerHTML = 'WASM';
      };
    }, 10);
  }

  function updateModel() {
    faceDetectModel.innerHTML = currentDetectionModel.modelName;
    emotionModel.innerHTML = currentEmotionModel.modelName;
  }

  function changeModel(newModel, modelType) {
    let modelClass;
    let currentModel;
    if (modelType === 'detect') {
      modelClass = faceDetector;
      currentModel = currentDetectionModel;
    } else {
      modelClass = emotionAnalysis;
      currentModel = currentEmotionModel;
    }
    if (currentModel.modelName === newModel.modelName) {
      return;
    }
    streaming = false;
    modelClass.deleteAll();
    removeAlertElement();
    setTimeout(async function() {
      try {
        if (modelType === 'detect') {
          progressLabel.innerHTML = 'Loading Face Detection Model:';
          progressContainer.style.display = "inline";
          faceDetectModel.innerHTML = 'Setting...';
        } else {
          progressLabel.innerHTML = 'Loading Emotion Analysis Model:';
          progressContainer.style.display = "inline";
          emotionModel.innerHTML = 'Setting...';
        }
        await modelClass.loadModel(newModel);
        await modelClass.init(currentBackend, currentPrefer);
        if (modelType === 'detect') {
          currentDetectionModel = newModel;
        } else {
          currentEmotionModel = newModel;
        }
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          Predict(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }
      catch(e) {
        let backend = currentBackend;
        if (currentBackend === 'WebML') {
          backend = 'WebML/' + getNativeAPI(currentPrefer);
        }
        console.warn(`Currently ${newModel.modelName} doesn't support ${backend} backend`);
        console.error(e);
        showAlert(backend, [currentDetectionModel.modelName, currentEmotionModel.modelName]);
        updateModel();
        modelClass.loadModel(currentModel);
      };
    }, 10);
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    emotionAnalysis.deleteAll();
    removeAlertElement();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(async function() {
      try {
        await faceDetector.init(currentBackend, newPrefer);
        await emotionAnalysis.init(currentBackend, newPrefer);
        currentPrefer = newPrefer;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          Predict(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }
      catch(e) {
        let currentBackend = 'WebML/' + getNativeAPI(currentPrefer);
        let nextBackend = 'WebML/' + getNativeAPI(newPrefer);
        console.warn(`Failed to change backend ${nextBackend}, switch back to ${currentBackend}`);
        console.error(e);
        changePrefer(currentPrefer, true);
        showAlert(nextBackend, [currentDetectionModel.modelName, currentEmotionModel.modelName]);
        updatePrefer();
        updateBackend();
      };
    }, 10);
  }

  function fileExists(url) {
    var exists;
    $.ajax({
      url:url,
      async:false,
      type:'HEAD',
      error:function() { exists = 0; },
      success:function() { exists = 1; }
    });
    if (exists === 1) {
      return true;
    } else {
      return false;
    }
  }

  function updateProgress(ev) {
    if (ev.lengthComputable) {
      let percentComplete = ev.loaded / ev.total * 100;
      percentComplete = percentComplete.toFixed(0);
      progressBar.style = `width: ${percentComplete}%`;
      progressBar.innerHTML = `${percentComplete}%`;
      if (ev.loaded === ev.total) {
        progressContainer.style.display = "none";
        progressBar.style = `width: 0%`;
        progressBar.innerHTML = `0%`;
      }
    }
  }

  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    }
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL');
    }
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    }
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WASM';
    }
  }

  // register face detection models
  for (let model of faceDetectionModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model, 'detect'));
    $('.available-detection-models').append(dropdownBtn);
    if (!currentDetectionModel) {
      currentDetectionModel = model;
    }
  }

  // register landmark detection model
  for (let model of emotionAnalysisModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model, 'emotion'));
    $('.available-landmark-models').append(dropdownBtn);
    if (!currentEmotionModel) {
      currentEmotionModel = model;
    }
  }

  // register prefers
  if (currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let sustained = $('<button class="dropdown-item"/>')
      .text('SUSTAINED_SPEED')
      .click(_ => changePrefer('sustained'));
    $('.preference').append(sustained);
    if (currentOS === 'Android') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item"/>')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Windows' || currentOS === 'Linux') {
      let fast = $('<button class="dropdown-item" disabled />')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Mac OS') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    }
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  // image or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    imageElement.onload = function() {
      Predict(imageElement);
    }

    try{
      progressLabel.innerHTML = 'Loading Landmark Detection Model:';
      await faceDetector.loadModel(currentDetectionModel);
      progressContainer.style.display = "inline";
      progressLabel.innerHTML = 'Loading Face Detection Model:';
      await emotionAnalysis.loadModel(currentEmotionModel);

      await faceDetector.init(currentBackend, currentPrefer);
      await emotionAnalysis.init(currentBackend, currentPrefer);
      
      updateBackend();
      updateModel();
      updatePrefer();
      Predict(imageElement);
      button.setAttribute('class', 'btn btn-primary');
      input.removeAttribute('disabled');
    }
    catch(e) {
      console.warn(`Failed to init ${currentBackend}, try to use WASM`);
      console.error(e);
      showAlert(currentBackend, [currentDetectionModel.modelName, currentEmotionModel.modelName]);
      changeBackend('WASM');
    };
  } else {
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then(async function (stream) {
      video.srcObject = stream;
      try {
        progressLabel.innerHTML = 'Loading Landmark Detection Model:';
        await faceDetector.loadModel(currentDetectionModel);
        progressContainer.style.display = "inline";
        progressLabel.innerHTML = 'Loading Face Detection Model:';
        await emotionAnalysis.loadModel(currentEmotionModel);

        await faceDetector.init(currentBackend, currentPrefer);
        await emotionAnalysis.init(currentBackend, currentPrefer);

        updateBackend();
        updateModel();
        updatePrefer();
        streaming = true;
        startPredict();
      }
      catch(e) {
        console.warn(`Failed to init ${currentBackend}, try to use WASM`);
        console.error(e);
        showAlert(currentBackend, [currentDetectionModel.modelName, currentEmotionModel.modelName]);
        changeBackend('WASM');
      };
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    function startPredict() {
      if (streaming) {
        videoElement.width = videoElement.videoWidth;
        videoElement.height = videoElement.videoHeight;
        stats.begin();
        Predict(videoElement);
        stats.end();
        setTimeout(startPredict, 0);
      }
    }
  }

  async function Predict(imageSource) {
    let detectResult = await faceDetector.getFaceBoxes(imageSource);
    let faceBoxes = detectResult.boxes;
    let time = parseFloat(detectResult.time);
    let keyPoints = [];
    for (let i = 0; i < faceBoxes.length; ++i) {
      let landmarkResult = await emotionAnalysis.predict(imageSource, faceBoxes[i]);
      keyPoints.push(landmarkResult.keyPoints.slice());
      time += parseFloat(landmarkResult.time);
    }
    let classes = getTopClasses(keyPoints, emotionAnalysis.labels, 1);
    drawFaceBoxes(imageSource, canvasShowElement, faceBoxes, classes);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${time.toFixed(2)} </em>ms`;
  }
}