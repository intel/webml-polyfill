function getSearchParams_prefer() {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.has('prefer') ? searchParams.get('prefer') : '';
}

function getSearchParams_backend() {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.has('b') ? searchParams.get('b') : '';
}

function getSearchParams_model() {
  let searchParams = new URLSearchParams(location.search);
  if (searchParams.has('m') && searchParams.has('t')) {
    return searchParams.get('m') + '_' + searchParams.get('t');
  } else {
    return '';
  }
}

const videoElement = document.getElementById('video');
const imageElement = document.getElementById('image');
const inputElement = document.getElementById('input');
const canvasElement = document.getElementById('canvas');
const progressBar = document.getElementById('progressBar');

let currentBackend = getSearchParams_backend();
let currentModel = getSearchParams_model();
let currentPrefer = getSearchParams_prefer();
let streaming = false;
let stats = new Stats();
let track;

let utils = new Utils(canvasElement);
utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

function showAlert(error) {
  let div = document.createElement('div');
  // div.setAttribute('id', 'backendAlert');
  div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = `<strong>${error}</strong>`;
  div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
  let container = document.getElementById('container');
  container.insertBefore(div, container.firstElementChild);
}

function updateProgress(ev) {
  if (ev.lengthComputable) {
    let percentComplete = ev.loaded / ev.total * 100;
    percentComplete = percentComplete.toFixed(0);
    progressBar.style = `width: ${percentComplete}%`;
    progressBar.innerHTML = `Loading Model: ${percentComplete}%`;
    updateLoading(percentComplete);
  }
}

function updateResult(result) {
  try {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.time} ms</span>`;
  } catch(e) {
    console.log(e);
  }
  try {
    console.log(`Classes: `);
    result.classes.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  }
  catch(e) {
    console.log(e);
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
for (let model of imageClassificationModels) {
  if (currentModel == model.modelName) {
    utils.changeModelParam(model)
  }
}

// register prefers
if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
  if (!currentPrefer) {
    currentPrefer = "sustained";
  }
}

async function startPredictCamera() {
  if (streaming) {
    try {
      stats.begin();
      let ret = await utils.predict(videoElement);
      updateResult(ret);
      stats.end();
      setTimeout(startPredictCamera, 0);
    } catch (e) {
      showAlert(e);
      showError();
    }
  }
}

async function utilsPredict(imageElement, backend, prefer) {
  streaming = false;
  // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
  if(track) {
    track.stop();
  }
  showProgress();
  try {
    utils.deleteAll();
  } catch (e) {
     console.log('utils.deleteAll(): ' + e);
  }
  try {
    await utils.init(backend, prefer);
    let ret = await utils.predict(imageElement);
    showResults();
    updateResult(ret);
  }
  catch (e) {
    showAlert(e);
    showError();
  }
}

async function utilsPredictCamera(backend, prefer) {
  streaming = true;
  showProgress();
  try {
    await utils.init(backend, prefer);
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: "environment" } });
    video.srcObject = stream;
    track = stream.getTracks()[0];
    startPredictCamera();
    showResults();
  } 
  catch (e) {
    showAlert(e);
    showError();
  }
}

async function updateScenario(camera, backend, prefer) {
  streaming = false;
  try {
    utils.deleteAll();
  } catch (e) {
     console.log('utils.deleteAll(): ' + e);
  }
  if (!camera) {
    utilsPredict(imageElement, backend, prefer);
  } else {
    utilsPredictCamera(backend, prefer);
  }
}

async function main(camera) {
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    imageElement.onload = function () {
      utilsPredict(imageElement, currentBackend, currentPrefer);
    }

    utilsPredict(imageElement, currentBackend, currentPrefer);
  } else {
    utilsPredictCamera(currentBackend, currentPrefer);
  }
}
