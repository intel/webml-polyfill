const deeplab513 = {
  modelName: 'DeepLab 513',
  modelFile: './model/deeplab_mobilenetv2_513.tflite',
  labelsFile: './model/labels.txt',
  inputSize: [513, 513, 3],
  outputSize: [513, 513, 21],
};

const deeplab224 = {
  modelName: 'DeepLab 224',
  modelFile: './model/deeplab_mobilenetv2_224.tflite',
  labelsFile: './model/labels.txt',
  inputSize: [224, 224, 3],
  outputSize: [224, 224, 21],
};

function main(camera) {

  const availableModels = [
    deeplab224,
    deeplab513,
  ];
  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const segMapCanvas = document.getElementsByClassName('seg-map')[0];
  let currentBackend = '';
  let currentModel = '';
  let streaming = false;

  let utils = new Utils();
  // register updateProgress function if progressBar element exist
  utils.progressCallback = updateProgress;


  function checkPreferParam() {
    if (getOS() === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Failed to setup ${backend} backend.</strong>`;
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
    let backendAlertElem = document.getElementById('backendAlert');
    if (backendAlertElem !== null) {
      backendAlertElem.remove();
    }
    let preferAlertElem = document.getElementById('preferAlert');
    if (preferAlertElem !== null) {
      preferAlertElem.remove();
    }
  }

  function updateBackend() {
    currentBackend = utils.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend).then(() => {
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
      });
    }, 10);
  }

  function clearCanvas() {
    const context = segMapCanvas.getContext('2d');
    context.clearRect(0, 0, segMapCanvas.width, segMapCanvas.height);
  }

  function updateModel() {
    selectModel.innerHTML = currentModel;
  }

  function changeModel(newModel) {
    if (currentModel === newModel.modelName) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    utils.changeModelParam(newModel);
    clearCanvas();
    progressContainer.style.display = "inline";
    selectModel.innerHTML = 'Setting...';
    if (camera) {
      video.style.maxHeight = newModel.inputSize[0] + 'px';
    } else {
      image.style.maxHeight = newModel.inputSize[0] + 'px';
    }
    $('.image-wrapper').css({
      'max-height': newModel.inputSize[0] + 'px',
      'max-width': newModel.inputSize[0] + 'px',
    });
    setTimeout(() => {
      utils.init(utils.model._backend).then(() => {
        currentModel = newModel.modelName;
        updateModel();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      });
    }, 10);
  }

  function _fileExists(url) {
    var exists;
    $.ajax({
      url: url,
      async: false,
      type: 'HEAD',
      error: function() { exists = 0; },
      success: function() { exists = 1; }
    });
    return exists === 1;
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

  function updateResult(result) {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${result.time} </em>ms`;

    let start = performance.now();
    drawSegMap(segMapCanvas, result.segMap);
    console.log(`[Main] Draw time: ${(performance.now() - start).toFixed(2)} ms`);
  }

  // register backends
  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    };
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function (e) {
      removeAlertElement();
      changeBackend('WebGL');
    };
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function (e) {
      removeAlertElement();
      changeBackend('WASM');
    };
  }

  // register models
  for (let model of availableModels) {
    if (!_fileExists(model.modelFile))
      continue;
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model));
    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      if (camera) {
        video.style.maxHeight = model.inputSize[0] + 'px';
      } else {
        image.style.maxHeight = model.inputSize[0] + 'px';
      }
      $('.image-wrapper').css({
        'max-height': model.inputSize[0] + 'px',
        'max-width': model.inputSize[0] + 'px',
      });
      currentModel = model.modelName;
    }
  }

  // picture or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    imageElement.onload = function () {
      utils.predict(imageElement).then(ret => updateResult(ret));
    };

    utils.init('WebGL').then(() => {
      updateBackend();
      updateModel();
      utils.predict(imageElement).then(ret => updateResult(ret));
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(utils.model._backend);
      changeBackend('WASM');
    });
  } else {
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
      video.srcObject = stream;
      utils.init('WebGL').then(() => {
        updateBackend();
        updateModel();
        streaming = true;
        startPredict();
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend);
        changeBackend('WASM');
      });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    function startPredict() {
      if (streaming) {
        stats.begin();
        utils.predict(videoElement).then(ret => updateResult(ret)).then(() => {
          stats.end();
          setTimeout(startPredict, 0);
        });
      }
    }
  }
}