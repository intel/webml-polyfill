const targetImageElement = document.getElementById('targetImage');
const searchImageElement = document.getElementById('searchImage');
const cameraImageElement = document.getElementById('cameraImage');
const targetInputElement = document.getElementById('targetInput');
const searchInputElement = document.getElementById('searchInput');
const cameraInputElement = document.getElementById('cameraImageInput');
const targetCanvasShowElement = document.getElementById('targetCanvasShow');
const searchCanvasShowElement = document.getElementById('searchCanvasShow');
const cameraImageShowElement = document.getElementById('cameraImageShow');
const cameraShowElement = document.getElementById('cameraShow');
const detecorCanvasElement = document.getElementById('detecorCanvas');
const recognitionCanvasElement = document.getElementById('recognitionCanvas');

let front = true;
let currentDetecorModel = null;
let currentRecognitionModel = null;

const detectionModels = faceDetectionModels.map((m) => m.modelId);
const recognitionModels = faceRecognitionModels.map((m) => m.modelId);
const faceDetector = new FaceDetecor(detecorCanvas);
const faceRecognition = new Utils(recognitionCanvasElement);

faceDetector.updateProgress = updateProgress;
faceRecognition.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

const updateResult = (result) => {
  try {
    console.log(`Inference time: ${result.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.toFixed(2)} ms</span>`;
  } catch(e) {
    console.log(e);
  }
}

const predict = async(targetSource, searchSource, targetShow, searchShow, embeddings=null) => {
  let targetDetectResult, targetFaceBoxes, targetTime,
      targetRecogniseResult, targetEmbeddings;
  if (embeddings == null) {
    targetDetectResult = await faceDetector.getFaceBoxes(targetSource);
    targetFaceBoxes = faceRecognition.getBoxs(targetDetectResult.boxes);
    targetRecogniseResult = await faceRecognition.predict(targetSource, targetFaceBoxes);
    targetEmbeddings = targetRecogniseResult.embedding;
    targetTime = parseFloat(targetDetectResult.time) + parseFloat(targetRecogniseResult.time);

    let targetClasses = new Array();
    for (let i in targetEmbeddings) targetClasses.push(parseInt(i) + 1);
    drawFaceBoxes(targetSource, targetShow, targetFaceBoxes, targetClasses);
  } else {
    targetEmbeddings = embeddings;
    targetTime = 0.0;
  }

  let searchDetectResult = await faceDetector.getFaceBoxes(searchSource);
  let searchFaceBoxes = faceRecognition.getBoxs(searchDetectResult.boxes);
  let searchRecogniseResult = await faceRecognition.predict(searchSource, searchFaceBoxes);
  let searchEmbeddings = searchRecogniseResult.embedding;
  let searchTime = parseFloat(searchDetectResult.time) + parseFloat(searchRecogniseResult.time);
  let searchClasses = faceRecognition.getClass(targetEmbeddings, searchEmbeddings);

  drawFaceBoxes(searchSource, searchShow, searchFaceBoxes, searchClasses);

  let runTime = parseFloat(targetTime) + parseFloat(searchTime);
  updateResult(runTime);

  return targetEmbeddings;
}

const utilsPredict = async (targetSource, searchSource) => {
  streaming = false;
  // Stop webcam opened by navigator.getUserMedia if user visits 'LIVE CAMERA' tab before
  if(track) {
    track.stop();
  }
  try {
    await showProgress('done', 'done', 'current');
    await predict(targetSource, searchSource, targetCanvasShowElement, searchCanvasShowElement);
    await showProgress('done', 'done', 'done');
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const startPredict = async (cameraImage, refresh=false, embeddings=null) => {
  if (streaming || refresh) {
    if (refresh && !streaming) streaming = true;
    videoElement.width = videoElement.videoWidth;
    videoElement.height = videoElement.videoHeight;
    stats.begin();
    let embedding = await predict(cameraImage, videoElement, cameraImageShowElement, cameraShowElement, embeddings);
    stats.end();
    setTimeout(startPredict, 0, cameraImage, false, embedding);
  }
}

const utilsPredictCamera = async (cameraImage) => {
  streaming = false;
  try {
    let stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: (front ? 'user' : 'environment') } });
    video.srcObject = stream;
    track = stream.getTracks()[0];
    await showProgress('done', 'done', 'current');
    startPredict(cameraImage, true);
    await showProgress('done', 'done', 'done');
    showResults();
  }
  catch (e) {
    errorHandler(e);
  }
}

const predictPath = (camera) => {
  (!camera) ? utilsPredict(targetImageElement, searchImageElement) : utilsPredictCamera(cameraImageElement);
}

const utilsInit = async (backend, prefer) => {
  // return immediately if model, backend, prefer are all unchanged
  let detectorInit = await faceDetector.init(backend, prefer);
  let recognitionInit = await faceRecognition.init(backend, prefer);
  if (detectorInit == 'NOT_LOADED' || recognitionInit == 'NOT_LOADED') {
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
    faceRecognition.initialized = false;
  }
  streaming = false;
  try {
    faceDetector.deleteAll();
    searchFaceDetector.deleteAll();
  } catch (e) { }
  logConfig();
  await showProgress('done', 'current', 'pending');
  try {
    getOffloadOps(currentBackend, currentPrefer);
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(faceRecognition.getSubgraphsSummary());
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
    faceRecognition.getRequiredOps()
  ]);
  return opsOfAllModels.reduce((a, b) => new Set([...a, ...b]));
}

const main = async (camera = false) => {
  streaming = false;

  if (currentModel !== 'undefined') {
    let currentModelArray = currentModel.split(/\s|\+/);
    for (let curtModel of currentModelArray) {
      if (recognitionModels.includes(curtModel)) {
        if (curtModel !== currentRecognitionModel) {
          try {
            faceRecognition.deleteAll();
          } catch (e) { }
        }

        currentRecognitionModel = curtModel;
      } else if (detectionModels.includes(curtModel)) {
        if (curtModel !== currentDetecorModel) {
          try {
            faceDetector.deleteAll();
          } catch (e) { }
        }

        currentDetecorModel = curtModel;
      } else {
        throw new Error(`Unknown model: ${curtModel}`);
      }
    }
  }

  logConfig();
  await showProgress('current', 'pending', 'pending');
  try {
    let faceDetecorModel = getModelById(currentDetecorModel);
    await faceDetector.loadModel(faceDetecorModel);
    let faceRecognitionModel = getModelById(currentRecognitionModel);
    await faceRecognition.loadModel(faceRecognitionModel);
    getOffloadOps(currentBackend, currentPrefer);
    await showProgress('done', 'current', 'pending');
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(faceRecognition.getSubgraphsSummary());
    predictPath(camera);
  } catch (e) {
    errorHandler(e);
  }
}
