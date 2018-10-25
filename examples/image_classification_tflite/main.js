var INPUT_SIZE, OUTPUT_TENSOR_SIZE, MODEL_FILE, LABELS_FILE;

function main() {
  let utils = new Utils();
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const selectModel = document.getElementById('selectModel');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  var progressContainer = document.getElementById('progressContainer');
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
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend).then(() => {
        updateBackend();
        utils.predict(imageElement);
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
    utils.deleteAll();
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
        utils.predict(imageElement);
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

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  imageElement.onload = function() {
    utils.predict(imageElement);
  }

  utils.inputTensor = new Float32Array(INPUT_SIZE * INPUT_SIZE * 3);
  utils.outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
  canvasElement.width = INPUT_SIZE;
  canvasElement.height = INPUT_SIZE;

  utils.init().then(() => {
    updateBackend();
    selectModel.innerHTML = currentModel;
    utils.predict(imageElement);
    button.setAttribute('class', 'btn btn-primary');
    input.removeAttribute('disabled');
  }).catch((e) => {
    console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
    console.error(e);
    showAlert(utils.model._backend);
    changeBackend('WASM');
  });
}
