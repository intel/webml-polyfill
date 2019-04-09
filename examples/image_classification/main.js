const canvasElement = document.getElementById('canvas');

let utils = new Utils(canvasElement);
utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

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

const startPredictCamera = async () => {
  if (streaming) {
    try {
      stats.begin();
      let ret = await utils.predict(videoElement);
      updateResult(ret);
      stats.end();
      setTimeout(startPredictCamera, 0);
    } catch (e) {
      errorHandler(e);
    }
  }
}

const utilsPredict = async (imageElement, backend, prefer) => {
  streaming = false;
  // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
  if (track) {
    track.stop();
  }
  await showProgress('Image inferencing ...');
  try {
    let ret = await utils.predict(imageElement);
    showResults();
    updateResult(ret);
  }
  catch (e) {
    errorHandler(e);
  }
}

const utilsPredictCamera = async (backend, prefer) => {
  streaming = true;
  await showProgress('Camera inferencing ...');
  try {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: 'environment' } });
    video.srcObject = stream;
    track = stream.getTracks()[0];
    startPredictCamera();
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const predictPath = (camera) => {
  (!camera) ? utilsPredict(imageElement, currentBackend, currentPrefer) : utilsPredictCamera(currentBackend, currentPrefer);
}

const utilsInit = async (backend, prefer) => {
  // return immediately if model, backend, prefer are all unchanged
  let init = await utils.init(backend, prefer);
  if (init == 'NOT_LOADED') {
    return;
  }
}

const updateScenario = async (camera = false) => {
  streaming = false;
  logConfig();
  predictPath(camera);
}

const updateBackend = async (camera = false, force = false) => {
  if (force) {
    utils.initialized = false;
  }
  streaming = false;
  try { utils.deleteAll(); } catch (e) { }
  logConfig();
  await showProgress('Updating backend ...');
  try {
    getOffloadOps(currentBackend, currentPrefer);
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
    predictPath(camera);
  }
  catch (e) {
    errorHandler(e);
  }
}

const main = async (camera = false) => {
  streaming = false;
  try { utils.deleteAll(); } catch (e) { }
  logConfig();
  await showProgress('Loading model ...');
  try {
    let model = imageClassificationModels.filter(f => f.modelFormatName == currentModel);
    await utils.loadModel(model[0]);
    getOffloadOps(currentBackend, currentPrefer);
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
  } catch (e) {
    errorHandler(e);
  }
  predictPath(camera);
}

