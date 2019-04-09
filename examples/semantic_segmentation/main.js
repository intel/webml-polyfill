const outputCanvas = document.getElementById('canvasvideo');

let currentTab = 'image';

let clippedSize = [];
let hoverPos = null;

const counterN = 20;
let counter = 0;
let inferTimeAcc = 0;
let drawTimeAcc = 0;

let renderer = new Renderer(outputCanvas);
renderer.setup();

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
    let init = await utils.init(currentBackend, currentPrefer);    
    if (init == 'NOT_LOADED') {
      return;
    }
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
  buttonUI(us === 'camera');
}

const predictPath = (camera) => {
  camera ? predictCamera() : predictAndDraw(imageElement, false);
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
    await utils.init(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
    predictPath(camera);
  }
  catch (e) {
    errorHandler(e);
  }
}

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
    await utils.loadModel(model[0]);
    getOffloadOps(currentBackend, currentPrefer);
    await utils.init(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
  } catch (e) {
    errorHandler(e);
  }
  predictPath(camera);
}
