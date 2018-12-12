const srgan_tflite = {
  modelName: 'SRGAN',
  inputSize: [96, 96, 3],
  outputSize: [384, 384, 3],
  scale: 4,
  modelFile: './model/srgan_96_4.tflite',
};

const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'MPS',
  'fast': 'BNNS',
};

function main(camera) {
  const inputCanvas = document.getElementById('inputCanvas');
  const outputCanvas = document.getElementById('outputCanvas');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const selectPrefer = document.getElementById('selectPrefer');

  let imageUrl = './image/image0.png';
  let currentBackend = '';
  let currentPrefer = '';
  let streaming = false;

  const utils = new Utils(srgan_tflite);
  utils.updateProgress = updateProgress; // register updateProgress function if progressBar element exist

  function checkPreferParam() {
    if (getOS() === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WebGL.");
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
        updateBackend();
        if (!camera) {
          computeAndDraw(imageUrl);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
        updatePrefer();
        updateBackend();
      });
    }, 10);
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, newPrefer).then(() => {
        currentPrefer = newPrefer;
        updatePrefer();
        updateBackend();
        if (!camera) {
          computeAndDraw(imageUrl);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${preferMap[newPrefer]}, switch back to ${preferMap[currentPrefer]}`);
        console.error(e);
        showAlert(preferMap[newPrefer]);
        changePrefer(currentPrefer, true);
        updatePrefer();
        updateBackend();
      });
    }, 10);
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
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

  // register prefers
  if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let MPS = $('<button class="dropdown-item"/>')
      .text('MPS')
      .click(_ => changePrefer(preferMap['MPS']));
    $('.preference').append(MPS);
    let BNNS = $('<button class="dropdown-item"/>')
      .text('BNNS')
      .click(_ => changePrefer(preferMap['BNNS']));
    $('.preference').append(BNNS);
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  // load img, predict, and draw result
  async function computeAndDraw(imageUrl) {
    await utils.loadImage(imageUrl, inputCanvas);
    utils.predict(inputCanvas).then(ret => updateResult(ret));

    let start = performance.now();
    utils.drawResult(outputCanvas);
    let elapsed = performance.now() - start;
    console.log(`draw time: ${elapsed.toFixed(2)} ms`);
  }

  // image or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageUrl = URL.createObjectURL(files[0]);
        computeAndDraw(imageUrl);
      }
    }, false);

    utils.init(currentBackend, currentPrefer).then(() => {
      updateBackend();
      updatePrefer();
      computeAndDraw(imageUrl);
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WebGL`);
      console.error(e);
      showAlert(utils.model._backend);
      changeBackend('WebGL');
    });
  } else {
    /*
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
      video.srcObject = stream;
      utils.init(currentBackend, currentPrefer).then(() => {
        updateBackend();
        updatePrefer();
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
    */
  }
}
