const canvasElement = document.getElementById('canvas');
const canvasShowElement = document.getElementById('canvasshow');
const canvasElement1 = document.getElementById('canvas1');

let front = true;
let currentLandmarkModel = 'face_landmark_tflite';

let faceDetector = new FaceDetecor(canvasElement);
let landmarkDetector = new Utils(canvasElement1);
faceDetector.updateProgress = updateProgress;
landmarkDetector.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

const updateResult = (result) => {
  try {
    console.log(`Inference time: ${result.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.toFixed(2)} ms</span>`;
  } catch(e) {
    console.log(e);
  }
}

const predict = async(source) => {
  let detectResult = await faceDetector.getFaceBoxes(source);
  let faceBoxes = detectResult.boxes;
  let time = parseFloat(detectResult.time);
  let keyPoints = [];
  for (let i = 0; i < faceBoxes.length; ++i) {
    let landmarkResult = await landmarkDetector.predict(source, faceBoxes[i]);
    keyPoints.push(landmarkResult.keyPoints.slice());
    time += parseFloat(landmarkResult.time);
  }
  drawFaceBoxes(source, canvasShowElement, faceBoxes);
  drawKeyPoints(source, canvasShowElement, keyPoints, faceBoxes);
  updateResult(time);
}

const utilsPredict = async (source) => {
  streaming = false;
  // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
  if(track) {
    track.stop();
  }
  try {
    await showProgress('done', 'done', 'current');
    await predict(source);
    await showProgress('done', 'done', 'done');
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const startPredict = async () => {
  if (streaming) {
    videoElement.width = videoElement.videoWidth;
    videoElement.height = videoElement.videoHeight;
    stats.begin();
    await predict(videoElement);
    stats.end();
    setTimeout(startPredict, 0);
  }
}

const utilsPredictCamera = async () => {
  streaming = true;
  try {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: (front ? 'user' : 'environment') } });
    video.srcObject = stream;
    track = stream.getTracks()[0];
    await showProgress('done', 'done', 'current');
    startPredict();
    await showProgress('done', 'done', 'done');
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const predictPath = (camera) => {
  (!camera) ? utilsPredict(imageElement) : utilsPredictCamera();
}

const utilsInit = async (backend, prefer) => {
  // return immediately if model, backend, prefer are all unchanged
  let init = await faceDetector.init(backend, prefer);
  let initlandmark = await landmarkDetector.init(backend, prefer);  
  if (init == 'NOT_LOADED' || initlandmark == 'NOT_LOADED') {
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
    faceDetector.initialized = false;
    landmarkDetector.initialized = false;
  }
  streaming = false;
  try { 
    landmarkDetector.deleteAll();
    faceDetector.deleteAll();
  } catch (e) { }
  logConfig();
  await showProgress('done', 'current', 'pending');
  try {
    getOffloadOps(currentBackend, currentPrefer);
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(landmarkDetector.getSubgraphsSummary());
    predictPath(camera);
  }
  catch (e) {
    errorHandler(e);
  }
}

// override `requiredOps` in main.common.js
requiredOps = async() => {
  const opsOfAllModels = await Promise.all([
    faceDetector.getRequiredOps(),
    landmarkDetector.getRequiredOps()
  ]);
  return opsOfAllModels.reduce((a, b) => new Set([...a, ...b]));
}

const main = async (camera = false) => {
  streaming = false;
  try { 
    // landmarkDetector.deleteAll();
    faceDetector.deleteAll();
  } catch (e) { }
  logConfig();
  await showProgress('current', 'pending', 'pending');
  try {
    let model = getModelById(currentModel);
    await faceDetector.loadModel(model);
    let landmarkmodel = getModelById(currentLandmarkModel);
    await landmarkDetector.loadModel(landmarkmodel);
    getOffloadOps(currentBackend, currentPrefer);
    await showProgress('done', 'current', 'pending');
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(landmarkDetector.getSubgraphsSummary());
    predictPath(camera);
  } catch (e) {
    errorHandler(e);
  }
}

