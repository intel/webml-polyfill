let inputElement = null;
let pickBtnElement = null;
let canvasElement = null;
let showCanvasElement = null;
let segCanvas = null;
let superCanvas = null;
let bkImageSrc = null;
let imageElement = document.getElementById('image');
let modelElement = document.getElementById('modelName');
let preferDivElement = document.getElementById('preferDiv');
let preferSelectElement = document.getElementById('preferSelect');

function getPreferString() {
  return preferSelectElement.options[preferSelectElement.selectedIndex].value;
}

function getSelectedOps() {
  return getPreferString() === 'none' ? new Set() : new Set(
    Array.from(
      document.querySelectorAll('input[name=supportedOp]:checked')).map(
        x => parseInt(x.value)));
}

function updateOpsSelect() {
  const backend = JSON.parse(configurations.value).backend;
  const prefer = preferSelect.value;
  if (backend !== 'WebNN' && prefer !== 'none') {
    // hybrid mode
    document.getElementById('supported-ops-select').style.visibility = 'visible';
  } else if (backend === 'WebNN' && prefer === 'none') {
    throw new Error('No backend selected');
  } else {
    // solo mode
    document.getElementById('supported-ops-select').style.visibility = 'hidden';
  }
  supportedOps = getSelectedOps();
}

async function setSuperImageUI(modelClass, modelName) {
  if (modelClass === 'super_resolution') {
    imageElement.parentNode.setAttribute('align', '');
    imageElement.style.width = '300px';
    imageElement.style.height = '300px';
    let srCanvas = document.createElement('canvas');
    let srCtx = srCanvas.getContext('2d');
    switch (modelName) {
      case 'SRGAN 96x4 (TFLite)':
        srCanvas.width = 96;
        srCanvas.height = 96;
        break;
      case 'SRGAN 128x4 (TFLite)':
        srCanvas.width = 128;
        srCanvas.height = 128;
        break;
      default:
        srCanvas.width = 96;
        srCanvas.height = 96;
    }
    let imageBytes = await loadImage(imageElement.src);
    srCtx.drawImage(imageBytes, 0, 0, srCanvas.width, srCanvas.height);
    imageElement.src = srCanvas.toDataURL();
  } else {
    imageElement.parentNode.setAttribute('align', 'center');
    imageElement.style.width = null;
    imageElement.style.height = null;
  }
}

function setImageSrc() {
  bkImageSrc = null;
  let inputFile = document.getElementById('input').files[0];
  let modelClass = modelElement.options[modelElement.selectedIndex].className;
  let modelName = modelElement.options[modelElement.selectedIndex].modelName;
  if (inputFile !== undefined) {
    imageElement.src = URL.createObjectURL(inputFile);
  } else {
    switch (modelClass) {
      case 'image_classification':
        imageElement.src = '../examples/image_classification/img/test.jpg';
        break;
      case 'skeleton_detection':
        imageElement.src = '../examples/skeleton_detection/img/download.png';
        break;
      case 'object_detection':
        imageElement.src = '../examples/object_detection/img/image1.jpg';
        break;
      case 'semantic_segmentation':
        imageElement.src = '../examples/semantic_segmentation/img/woman.jpg';
        break;
      case 'facial_landmark_detection':
        imageElement.src = '../examples/facial_landmark_detection/img/image1.jpg';
        break;
      case 'super_resolution':
        imageElement.src = '../examples/super_resolution/img/mushroom.png';
        break;
      case 'emotion_analysis':
        imageElement.src = '../examples/emotion_analysis/img/image1.jpg';
        break;
      default:
        imageElement.src = '../examples/image_classification/img/test.jpg';
    }
  }
  setSuperImageUI(modelClass, modelName);
}

document.addEventListener('DOMContentLoaded', () => {
  inputElement = document.getElementById('input');
  pickBtnElement = document.getElementById('pickButton');
  canvasElement = document.getElementById('canvas');
  showCanvasElement = document.getElementById('showCanvas');
  segCanvas = document.getElementById('segCanvas');
  superCanvas = document.getElementById('superCanvas');
  inputElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
    let files = e.target.files;
    if (files.length > 0) {
      bkImageSrc = null;
      imageElement.src = URL.createObjectURL(files[0]);
    }
    setImageSrc();
  }, false);
  modelElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
    setImageSrc();
  }, false);
  let configurationsElement = document.getElementById('configurations');
  configurationsElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
    setImageSrc();
    if (JSON.parse(e.target.value).backend === 'WebNN') {
      document.querySelector('#preferSelect option[value=none]').disabled = true;
    } else {
      document.querySelector('#preferSelect option[value=none]').disabled = false;
    }
    updateOpsSelect();
  }, false);
  let selectAllOpsElement = document.getElementById('selectAllOps');
  selectAllOpsElement.addEventListener('click', () => {
    document.querySelectorAll('input[name=supportedOp]').forEach((x) => {
      x.checked = true;
    });
  }, false);
  let uncheckAllOpsElement = document.getElementById('uncheckAllOps');
  uncheckAllOpsElement.addEventListener('click', () => {
    document.querySelectorAll('input[name=supportedOp]').forEach((x) => {
      x.checked = false;
    });
  }, false);
  let eagerModeElement = document.getElementById('eagerMode');
  eagerModeElement.addEventListener('change', (e) => {
    eager = e.target.checked;
  }, false);
  let preferSelectElement = document.getElementById('preferSelect');
  preferSelectElement.addEventListener('change', () => {
    updateOpsSelect();
    setImageSrc();
  }, false);
  let polyfillConfigurations = [{
    backend: 'WASM',
    modelName: '',
    modelClass: '',
    iterations: 0
  },
  {
    backend: 'WebGL',
    modelName: '',
    modelClass: '',
    iterations: 0
  }];
  let webnnConfigurations = [{
    backend: 'WebNN',
    modelName: '',
    modelClass: '',
    iterations: 0
  }];
  let configurations = [];
  configurations = configurations.concat(polyfillConfigurations, webnnConfigurations);
  for (let configuration of configurations) {
    let option = document.createElement('option');
    option.value = JSON.stringify(configuration);
    option.textContent = `${configuration.backend}`;
    document.querySelector('#configurations').appendChild(option);
  }
  let button = document.querySelector('#runButton');
  button.setAttribute('class', 'btn btn-primary');
  button.addEventListener('click', main);
});
