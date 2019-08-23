let inputElement = null;
let pickBtnElement = null;
let showCanvasElement = null;
let imageElement = document.getElementById('image');
let categoryElement = document.getElementById('categoryName');
let modelElement = document.getElementById('modelName');
let preferDivElement = document.getElementById('preferDiv');
let preferSelectElement = document.getElementById('preferSelect');
let runButton = document.querySelector('#runButton');

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

function showImage() {
  let ctx = showCanvasElement.getContext("2d");
  showCanvasElement.setAttribute("width", imageElement.width);
  showCanvasElement.setAttribute("height", imageElement.height);
  ctx.drawImage(imageElement, 0, 0);
}

function setImageSrc() {
  let inputFile = document.getElementById('input').files[0];
  let categoryId = categoryElement.options[categoryElement.selectedIndex].id;
  if (inputFile !== undefined) {
    imageElement.src = URL.createObjectURL(inputFile);
  } else {
    switch (categoryId) {
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
  imageElement.onload = function() {
    showImage();
  }
}

function setModelsOptions(category) {
  let modelsList = modelZoo[category];
  for (let model of modelsList) {
    let option = document.createElement('option');
    option.textContent = model.modelName;
    document.querySelector('#modelName').appendChild(option);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  inputElement = document.getElementById('input');
  pickBtnElement = document.getElementById('pickButton');
  showCanvasElement = document.getElementById('showCanvas');
  let curCategory = categoryElement.options[categoryElement.selectedIndex].className;
  setModelsOptions(curCategory);
  inputElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
    setImageSrc();
  }, false);
  categoryElement.addEventListener('change', (e) => {
    $('#modelName').empty();
    $('.labels-wrapper').empty();
    curCategory = categoryElement.options[categoryElement.selectedIndex].className;
    setModelsOptions(curCategory);
    setImageSrc();
  }, false);
  modelElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
  }, false);
  let configurationsElement = document.getElementById('configurations');
  configurationsElement.addEventListener('change', (e) => {
    $('.labels-wrapper').empty();
    setImageSrc();
    if (JSON.parse(e.target.value).backend === 'WebNN') {
      $("#preferSelect option[value='sustained']").prop("selected", true);
      document.querySelector('#preferSelect option[value=none]').disabled = true;
    } else {
      $("#preferSelect option[value='none']").prop("selected", true);
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
    category: '',
    iterations: 0
  },
  {
    backend: 'WebGL',
    modelName: '',
    category: '',
    iterations: 0
  }];
  let webnnConfigurations = [{
    backend: 'WebNN',
    modelName: '',
    category: '',
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
  runButton.addEventListener('click', main);
  showImage();
});
