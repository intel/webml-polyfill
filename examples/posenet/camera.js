const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const scaleCanvas = document.getElementById('scaleCanvas');
const scaleCtx = scaleCanvas.getContext('2d');
const backend = document.getElementById('backend');
const wasm = document.getElementById('wasm');
const webgl = document.getElementById('webgl');
const webml = document.getElementById('webml');
const selectPrefer = document.getElementById('selectPrefer');
let currentBackend = '';
let currentPrefer = '';

guiState.scoreThreshold = 0.15;

const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'MPS',
  'fast': 'BNNS',
};

const util = new Utils();
const videoWidth = 500;
const videoHeight = 500;
const inputSize = [1, videoWidth, videoHeight, 3];
const algorithm = gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);
let isMultiple = guiState.algorithm;
let streaming  = false;	
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  throw new Error(
    'Browser API navigator.mediaDevices.getUserMedia not available');
}
let stats = new Stats();
stats.dom.style.cssText = 'position:fixed;top:100px;left:10px;cursor:pointer;opacity:0.9;z-index:999';
stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
document.body.appendChild(stats.dom);
const mobile = isMobile();

algorithm.onFinishChange((algorithm) => {
  guiState.algorithm = algorithm;
});

scoreThreshold.onChange((scoreThreshold) => {
  guiState.scoreThreshold = parseFloat(scoreThreshold);
  util._minScore = guiState.scoreThreshold;
});

nmsRadius.onFinishChange((nmsRadius) => {
  guiState.multiPoseDetection.nmsRadius = parseInt(nmsRadius);
  util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
});

maxDetections.onFinishChange((maxDetections) => {
  guiState.multiPoseDetection.maxDetections = parseInt(maxDetections);
  util._maxDetection = guiState.multiPoseDetection.maxDetections;
});

model.onFinishChange((model) => {
  guiState.model = model;
  initModel();
});

outputStride.onFinishChange((outputStride) => {
  guiState.outputStride = parseInt(outputStride);
  initModel();
});

scaleFactor.onFinishChange((scaleFactor) => {
  guiState.scaleFactor = parseFloat(scaleFactor);
  initModel();
});

showPose.onChange((showPose) => {
  guiState.showPose = showPose;
});

showBoundingBox.onChange((showBoundingBox) => {
  guiState.showBoundingBox = showBoundingBox;
});

function updateBackend() {
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
  if (newBackend !== "WebML") {
    selectPrefer.style.display = 'none';
  } else {
    selectPrefer.style.display = 'inline';
  }
  streaming = false;
  util.deleteAll();
  backend.innerHTML = 'Setting...';
  setTimeout(() => {
    util.init(newBackend, currentPrefer, inputSize).then(() => {
      currentBackend = newBackend;
      updateBackend();
      updatePrefer();
      streaming = true;
      poseDetectionFrame();
    });
  }, 10);
}

function changePrefer(newPrefer, force) {
  if (currentPrefer === newPrefer && !force) {
    return;
  }
  streaming = false;
  util.deleteAll();
  selectPrefer.innerHTML = 'Setting...';
  setTimeout(() => {
    util.init(currentBackend, newPrefer, inputSize).then(() => {
      currentPrefer = newPrefer;
      updatePrefer();
      updateBackend();
      streaming = true;
      poseDetectionFrame();
    }).catch((e) => {
      console.warn(`Failed to change backend ${preferMap[newPrefer]}, switch back to ${preferMap[currentPrefer]}`);
      console.error(e);
      showAlert(preferMap[newPrefer]);
      changePrefer(currentPrefer, true);
      updatePrefer();
      updateBackend();
    });
  }, 10);
}

function updatePrefer() {
  selectPrefer.innerHTML = preferMap[currentPrefer];
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {facingMode: 'user'},
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    }
  });
}

async function loadVideo() {
  const videoElement = await setupCamera();
  videoElement.play();
  canvas.setAttribute("width", videoElement.videoWidth);
  canvas.setAttribute("height", videoElement.videoHeight);
  return videoElement;
}

async function initModel(first = false) {
  if (!first && !util.initialized) {
    console.warn('not initialized');
    return;
  }
  streaming = true;
  util.init(currentBackend, currentPrefer, inputSize).then(() => {
    updateBackend();
    updatePrefer();
    //streaming = true;
  }).catch((e) => {
    console.warn(`Failed to init ${util.model._backend}, try to use WASM`);
    console.error(e);
    showAlert(util.model._backend);
    changeBackend('WASM');
  });
}

function checkPreferParam() {
  if (getOS() === 'Mac OS') {
    let preferValue = getPreferParam();
    if (preferValue === 'invalid') {
      console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
      showPreferAlert();
    }
  }
}

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

async function main() {
  checkPreferParam();

  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    }
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL');
    }
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    }
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WASM';
    }
  }

  // register prefers
  if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let MPS = $('<button class="dropdown-item"/>')
      .text('MPS')
      .click(_ => changePrefer(preferMap['MPS']));
    $('.preference').append(MPS);
    let BNNS = $('<button class="dropdown-item"/>')
      .text('BNNS')
      .click(_ => changePrefer(preferMap['BNNS']));
    $('.preference').append(BNNS);
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  await loadVideo();
  initModel(true);
  poseDetectionFrame();
}
  
function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

function drawVideo(video, canvas, w, h) {
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-w, 0);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
}

async function poseDetectionFrame() {
  if (streaming) {
    if (util.initialized) {
      await predict(video);
    }
    setTimeout(poseDetectionFrame, 0);
  }
}

async function predict(video) {
  stats.begin();
  const start = performance.now();
  let type = guiState.algorithm == 'multi-pose' ? 'multi' : 'single';
  drawVideo(video, scaleCanvas, util.scaleWidth, util.scaleHeight);
  await util.predict(scaleCanvas, type);
  drawVideo(video, canvas, video.videoWidth, video.videoHeight);
  util.drawPoses(canvas, util.decodePose(type));
  const elapsed = performance.now() - start;
  const inferenceTimeElement = document.getElementById('inferenceTime');
  inferenceTimeElement.innerHTML = `Inference time: <em style="color:green;font-weight:bloder;">${elapsed.toFixed(2)} </em>ms`;
  stats.end();
}
