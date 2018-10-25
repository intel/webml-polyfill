var INPUT_SIZE, OUTPUT_TENSOR_SIZE, MODEL_FILE, LABELS_FILE;

function main() {
  let utils = new Utils();
  const videoElement = document.getElementById('video');
  let streaming = false;
  let changing = false;           // use to avoid the mismatch between canvas size(inputTensor size) and the old model's input oprand when changing model
  const backend = document.getElementById('backend');
  const selectModel = document.getElementById('selectModel');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  const progressContainer = document.getElementById('progressContainer');
  const Mobilenet_V1 = document.getElementById('Mobilenet_V1');
  const Mobilenet_V2 = document.getElementById('Mobilenet_V2');
  const Inception_V3 = document.getElementById('Inception_V3');
  const Squeezenet = document.getElementById('Squeezenet');

  let currentBackend = '';
  let currentModel = '';
  let chooseFirstModel = true;

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
    currentBackend = utils.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend) {
    if (currentBackend === newBackend) {
      return;
    }
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend).then(() => {
        updateBackend();
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend);
        changeBackend('WASM');
        backend.innerHTML = 'WASM';
      });
    }, 10);
  }

  function changeModel(newModel) {
    if (currentModel === newModel.MODEL_NAME) {
      return;
    }
    changing = true;
    streaming = false;
    INPUT_SIZE = newModel.INPUT_SIZE;
    OUTPUT_TENSOR_SIZE = newModel.OUTPUT_SIZE;
    MODEL_FILE = newModel.MODEL_FILE;
    LABELS_FILE = newModel.LABELS_FILE;
    utils.inputTensor = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
    utils.outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
    canvasElement.width = newModel.INPUT_SIZE;
    canvasElement.height = newModel.INPUT_SIZE;
    progressContainer.style.display = "inline";
    utils.tfModel = null;
    selectModel.innerHTML = 'Setting...';

    setTimeout(() => {
      utils.init(utils.model._backend).then(() => {
        currentModel = newModel.MODEL_NAME;
        selectModel.innerHTML = currentModel;
        streaming = true;
        changing =false;
        startPredict();
      });
    }, 10);
  }

  function fileExists(url) {
    var isExists;
    $.ajax({
      url:url,
      async:false,
      type:'HEAD',
      error:function() { isExists = 0; },
      success:function() { isExists = 1; }
    });
    if (isExists === 1) {
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
    }
  }

  if (nnPolyfill.supportWebGL2) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL2');
    }
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    }
  }

  //check if the model file exist and choose the first model
  if (fileExists(mobilenet_v1.MODEL_FILE)) {
    Mobilenet_V1.setAttribute('class', 'dropdown-item');
    Mobilenet_V1.onclick = function(e) {
      changeModel(mobilenet_v1);
    }
    if (chooseFirstModel) {
      INPUT_SIZE = mobilenet_v1.INPUT_SIZE;
      OUTPUT_TENSOR_SIZE = mobilenet_v1.OUTPUT_SIZE;
      MODEL_FILE = mobilenet_v1.MODEL_FILE;
      LABELS_FILE = mobilenet_v1.LABELS_FILE;
      currentModel = "Mobilenet_V1";
      chooseFirstModel = false;
    }
  }

  if (fileExists(mobilenet_v2.MODEL_FILE)) {
    Mobilenet_V2.setAttribute('class', 'dropdown-item');
    Mobilenet_V2.onclick = function(e) {
      changeModel(mobilenet_v2);
    }
    if (chooseFirstModel) {
      INPUT_SIZE = mobilenet_v2.INPUT_SIZE;
      OUTPUT_TENSOR_SIZE = mobilenet_v2.OUTPUT_SIZE;
      MODEL_FILE = mobilenet_v2.MODEL_FILE;
      LABELS_FILE = mobilenet_v2.LABELS_FILE;
      currentModel = "Mobilenet_V2";
      chooseFirstModel = false;
    }
  }
  
  if (fileExists(inception_v3.MODEL_FILE)) {
    Inception_V3.setAttribute('class', 'dropdown-item');
    Inception_V3.onclick = function(e) {
      changeModel(inception_v3);
    }
    if (chooseFirstModel) {
      INPUT_SIZE = inception_v3.INPUT_SIZE;
      OUTPUT_TENSOR_SIZE = inception_v3.OUTPUT_SIZE;
      MODEL_FILE = inception_v3.MODEL_FILE;
      LABELS_FILE = inception_v3.LABELS_FILE;
      currentModel = "Inception_V3";
      chooseFirstModel = false;
    }
  }
  
  if (fileExists(squeezenet.MODEL_FILE)) {
    Squeezenet.setAttribute('class', 'dropdown-item');
    Squeezenet.onclick = function(e) {
      changeModel(squeezenet);
    }
    if (chooseFirstModel) {
      INPUT_SIZE = squeezenet.INPUT_SIZE;
      OUTPUT_TENSOR_SIZE = squeezenet.OUTPUT_SIZE;
      MODEL_FILE = squeezenet.MODEL_FILE;
      LABELS_FILE = squeezenet.LABELS_FILE;
      currentModel = "Squeezenet";
      chooseFirstModel = false;
    }
  }

  utils.inputTensor = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  utils.outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
  canvasElement.width = INPUT_SIZE;
  canvasElement.height = INPUT_SIZE;

  let stats = new Stats();
  stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);

  navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
    video.srcObject = stream;
    utils.init().then(() => {
      updateBackend();
      selectModel.innerHTML = currentModel;
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
    if (!changing) {
      stats.begin();
      utils.predict(videoElement).then(() => {
        stats.end();
        if (streaming) {
          setTimeout(startPredict, 0);
        }
      });
    }
  }
}
