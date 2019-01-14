function main(camera) {
  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  const canvasShowElement = document.getElementById('canvasShow');
  const selectModel = document.getElementById('selectModel');
  const selectPrefer = document.getElementById('selectPrefer');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');

  let currentBackend = '';
  let currentModel;
  let currentPrefer = '';
  let streaming = false;

  let utils = new Utils(canvasElement, canvasShowElement);
  utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

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

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Currently ${backend} backend doesn't support SSD-MobileNet Model.</strong>`;
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
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend, currentPrefer).then(() => {
        currentBackend = newBackend;
        updateBackend();
        updateModel();
        updatePrefer();
        if (!camera) {
          utils.predict(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend, currentModel.modelName);
        changeBackend('WASM');
        updatePrefer();
        backend.innerHTML = 'WASM';
      });
    }, 10);
  }

  function updateModel() {
    selectModel.innerHTML = currentModel.modelName;
  }

  function changeModel(newModel) {
    if (currentModel.modelName === newModel.modelName) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    removeAlertElement();
    utils.changeModelParam(newModel);
    progressContainer.style.display = "inline";
    selectModel.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        currentModel = newModel;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        let backend = currentBackend;
        if (currentBackend === 'WebML') {
          backend = 'WebML/' + getNativeAPI(currentPrefer);
        }
        console.warn(`Currently ${newModel.modelName} doesn't support ${backend} backend`);
        console.error(e);
        showAlert(backend, newModel.modelName);
        updateModel();
        utils.changeModelParam(currentModel);
      });
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
          utils.predict(imageElement);
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
        showAlert(nextBackend, currentModel.modelName);
        updatePrefer();
        updateBackend();
      });
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

  // register models
  for (let model of objectDetectionModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model));
    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      currentModel = model;
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
      utils.predict(imageElement);
    }

    utils.init(currentBackend, currentPrefer).then(() => {
      updateBackend();
      updateModel();
      updatePrefer();
      utils.predict(imageElement);
      button.setAttribute('class', 'btn btn-primary');
      input.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(utils.model._backend, currentModel.modelName);
      changeBackend('WASM');
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
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend, currentModel.modelName);
        changeBackend('WASM');
      });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    function startPredict() {
      if (streaming) {
        stats.begin();
        utils.predict(videoElement).then(() => {
          stats.end();
          setTimeout(startPredict, 0);
        });
      }
    }
  }
}