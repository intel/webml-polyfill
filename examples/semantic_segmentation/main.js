const videoElement = document.getElementById('video');
const imageElement = document.getElementById('image');
const inputElement = document.getElementById('input');
const outputCanvas = document.getElementById('canvasvideo');
const progressBar = document.getElementById('progressBar');

let currentBackend = getSearchParamsBackend();
let currentModel = getSearchParamsModel();
let currentPrefer = getSearchParamsPrefer();
let currentTab = 'image';

let clippedSize = [];
let hoverPos = null;

let streaming = false;
let stats = new Stats();
let track;

const counterN = 20;
let counter = 0;
let inferTimeAcc = 0;
let drawTimeAcc = 0;

let renderer = new Renderer(outputCanvas);
renderer.setup();

const showAlert = (error) => {
  console.error(error);
  let div = document.createElement('div');
  // div.setAttribute('id', 'backendAlert');
  div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = `<strong>${error}</strong>`;
  div.innerHTML += `<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>`;
  let container = document.getElementById('container');
  container.insertBefore(div, container.firstElementChild);
}

const updateProgress = (ev) => {
  if (ev.lengthComputable) {
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    percentComplete = percentComplete.toFixed(0);
    progressBar.style = `width: ${percentComplete}%`;
    updateLoading(loadedSize.toFixed(1), totalSize.toFixed(1), percentComplete);
  }
}

let utils = new Utils();
utils.updateProgress = updateProgress;

const updateResult = (result) => {
  try {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.time} ms</span>`;
  } catch (e) {
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
  catch (e) {
    console.log(e);
  }
}

const logConfig = () => {
  console.log(`Model: ${currentModel}, Backend: ${currentBackend}, Prefer: ${currentPrefer}`);
}

const errorHandler = (e) => {
  showAlert(e);
  showError(null, null);
}

const startPredict = async () => {
  if (streaming) {
    try {
      stats.begin();
      await predictAndDraw(videoElement, true);
      stats.end();
      setTimeout(startPredict, 0);
    } catch (e) {
      errorHandler(e);
    }
  }
}

const predictCamera = async () => {
  try {
    streaming = true;
    // let res = utils.getFittedResolution(4 / 3);
    // setCamResolution(res);
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'user' } });
    videoElement.srcObject = stream;
    track = stream.getTracks()[0];
    showProgress('Inferencing ...');
    videoElement.onloadeddata = startPredict;
  } catch (e) {
    errorHandler(e);
  }
}

const predictAndDraw = async (source, camera = false) => {
  if (!camera) {
    streaming = false;
    if (track) track.stop();
  }
  clippedSize = utils.prepareInput(source);
  renderer.uploadNewTexture(source, clippedSize);
  let result = await utils.predict();
  let inferTime = result.time;
  console.log(`Inference time: ${inferTime.toFixed(2)} ms`);
  inferenceTime.innerHTML = `inference time: <span class='ir'>${inferTime.toFixed(2)} ms</span>`;
  renderer.drawOutputs(result.segMap)
  renderer.highlightHoverLabel(hoverPos);
  showResults();
}

const predictPath = (camera) => {
  camera ? predictCamera() : predictAndDraw(imageElement, currentBackend, currentPrefer, false);
}

const updateScenario = async (camera = false) => {
  streaming = false;
  logConfig();
  try { utils.deleteAll(); } catch (e) { }
  await showProgress('Inferencing ...');
  try {
    await utils.init(currentBackend, currentPrefer);
  }
  catch (e) {
    errorHandler(e);
  }
  predictPath(camera);
}

inputElement.addEventListener('change', (e) => {
  let files = e.target.files;
  if (files.length > 0) {
    imageElement.src = URL.createObjectURL(files[0]);
  }
}, false);

imageElement.addEventListener('load', () => {
  predictAndDraw(imageElement, false);
}, false);

const main = async (camera = false) => {
  if (currentModel === 'none_none') {
    errorHandler('No model selected');
    return;
  }
  console.log(currentModel)
  streaming = false;
  try { utils.deleteAll(); } catch (e) { }
  logConfig();
  showProgress('Loading model and initializing...');
  try {
    let model = semanticSegmentationModels.filter(f => f.modelFormatName == currentModel);
    utils.changeModelParam(model[0]);
    await utils.init(currentBackend, currentPrefer);
  } catch (e) {
    errorHandler(e);
  }
  predictPath(camera);
}
