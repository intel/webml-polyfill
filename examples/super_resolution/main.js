const fsrcnn_96_4 = {
  modelName: 'FSRCNN 96x4',
  inputSize: [96, 96, 3],
  outputSize: [384, 384, 3],
  scale: 4,
  modelFile: './model/fsrcnn_96_4.tflite'
};

const fsrcnn_128_4 = {
  modelName: 'FSRCNN 128x4',
  inputSize: [128, 128, 3],
  outputSize: [512, 512, 3],
  scale: 4,
  modelFile: './model/fsrcnn_128_4.tflite'
};

const srgan_96_4 = {
  modelName: 'SRGAN 96x4',
  inputSize: [96, 96, 3],
  outputSize: [384, 384, 3],
  scale: 4,
  modelFile: './model/srgan_96_4.tflite'
};

const srgan_128_4 = {
  modelName: 'SRGAN 128x4',
  inputSize: [128, 128, 3],
  outputSize: [512, 512, 3],
  scale: 4,
  modelFile: './model/srgan_128_4.tflite'
};

const availableModels = [
  fsrcnn_96_4,
  fsrcnn_128_4,
  srgan_96_4,
  srgan_128_4
];

function main(camera) {
  const imageElement = document.getElementById('image');
  const videoElement = document.getElementById('video');
  const inputCanvas = document.getElementById('inputCanvas');
  const outputCanvas = document.getElementById('outputCanvas');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const selectModel = document.getElementById('selectModel');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const selectPrefer = document.getElementById('selectPrefer');

  let currentBackend = '';
  let currentModel = '';
  let currentPrefer = '';
  let streaming = false;

  const utils = new Utils();
  utils.updateProgress = updateProgress; // register updateProgress function if progressBar element exist

  function checkPreferParam() {
    if (currentOS === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WebGL.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend, modelName) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Currently ${backend} doesn't support ${modelName} Model.</strong>`;
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

  function updateModel() {
    selectModel.innerHTML = currentModel;
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
  }

  function changeCanvasSize(newModel) {
    const inputSize = newModel.inputSize;
    const outputSize = newModel.outputSize;
    inputCanvas.width = inputSize[1];
    inputCanvas.height = inputSize[0];
    outputCanvas.width = outputSize[1];
    outputCanvas.height = outputSize[0];
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    streaming = false;
    if (newBackend !== "WebML") {
      selectPrefer.style.display = 'none';
    } else {
      selectPrefer.style.display = 'inline';
    }
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend, currentPrefer).then(() => {
        currentBackend = newBackend;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend, currentModel);
        changeBackend(currentBackend, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function changeModel(newModel) {
    if (currentModel === newModel.modelName) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    removeAlertElement();
    utils.changeModelParam(newModel);
    changeCanvasSize(newModel);
    progressContainer.style.display = "inline";
    currentPrefer = "sustained";
    selectModel.innerHTML = 'Setting...';
    currentModel = newModel.modelName;
    setTimeout(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      });
    }, 10);
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    removeAlertElement();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, newPrefer).then(() => {
        currentPrefer = newPrefer;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        let currentBackend = 'WebML/' + getNativeAPI(currentPrefer);
        let nextBackend = 'WebML/' + getNativeAPI(newPrefer);
        console.warn(`Failed to change backend ${nextBackend}, switch back to ${currentBackend}`);
        console.error(e);
        changePrefer(currentPrefer, true);
        showAlert(nextBackend, currentModel);
        updatePrefer();
        updateBackend();
      });
    }, 10);
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
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL');
    };
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    };
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WebGL';
    }
  }

  // register models
  for (let model of availableModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model));
    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      changeCanvasSize(model);
      currentModel = model.modelName;
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
    }  else if (currentOS === 'Mac OS') {
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

  async function predictAndDraw(imageElement) {
    utils.predict(imageElement).then(ret => updateResult(ret));
    let start = performance.now();
    utils.drawOutput(outputCanvas, imageElement);
    utils.drawInput(inputCanvas, imageElement);
    let elapsed = performance.now() - start;
    console.log(`draw time: ${elapsed.toFixed(2)} ms`);
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
      predictAndDraw(imageElement);
    }

    utils.init(currentBackend, currentPrefer).then(() => {
      updateBackend();
      updateModel();
      updatePrefer();
      predictAndDraw(imageElement);
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WebGL`);
      console.error(e);
      showAlert(utils.model._backend, currentModel);
      changeBackend('WebGL');
    });
  } else {
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
      video.srcObject = stream;
      utils.init(currentBackend, currentPrefer).then(() => {
        updateBackend();
        updateModel();
        updatePrefer();
        streaming = true;
        startPredict();
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WebGL`);
        console.error(e);
        showAlert(utils.model._backend, currentModel);
        changeBackend('WebGL');
      });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    function startPredict() {
      if (streaming) {
        stats.begin();
        predictAndDraw(videoElement).then(_ => {
          stats.end();
          setTimeout(startPredict, 0);
        });
      }
    }
  }
}
