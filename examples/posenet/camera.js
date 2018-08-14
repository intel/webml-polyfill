const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleCanvas = document.getElementById('scaleCanvas');
const scaleCtx = scaleCanvas.getContext('2d');
const backend = document.getElementById('backend');
const wasm = document.getElementById('wasm');
const webgl = document.getElementById('webgl');
const webml = document.getElementById('webml');
let currentBackend = '';

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

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video':{
      facingMode: 'user',
      width: mobile? undefined: videoWidth,
      height: mobile? undefined : videoHeight,
    },
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
  return videoElement;
}

async function detectPoseInRealTime(video) {
  async function poseDetectionFrame() {
    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-videoWidth, 0);
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    ctx.restore();
    await predict();
    algorithm.onChange((algorithm) => {
      guiState.algorithm = algorithm;
    });
    scoreThreshold.onChange((scoreThreshold) => {
      guiState.scoreThreshold = scoreThreshold;
      util._minScore = guiState.scoreThreshold;
    });
    nmsRadius.onChange((nmsRadius) => {
      guiState.multiPoseDetection.nmsRadius = nmsRadius;
      util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
    });
    maxDetections.onChange((maxDetections) => {
      guiState.multiPoseDetection.maxDetections = maxDetections;
      util._maxDetection = guiState.multiPoseDetection.maxDetections;
    });
    model.onChange((model) => {
      guiState.model = model;
      util._version = guiState.model;
      detectPoseInRealTime(video);
    });
    outputStride.onChange((outputStride) => {
      guiState.outputStride = outputStride;
      util._outputStride = guiState.outputStride;
      detectPoseInRealTime(video);
    });
    scaleFactor.onChange((scaleFactor) => {
      guiState.scaleFactor = scaleFactor; 
      util._scaleFactor = guiState.scaleFactor;
      detectPoseInRealTime(video);
    });
    setTimeout(poseDetectionFrame, 0);
  }
  function updateBackend() {
    currentBackend = util.model._backend;
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
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      util.init(newBackend, inputSize).then(() => {
        updateBackend();
      });
    }, 10);
  }

  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      changeBackend('WebML');
    }
  }

  if (nnPolyfill.supportWebGL2) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      changeBackend('WebGL2');
    }
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      changeBackend('WASM');
    }
  }

  if (currentBackend == '') {
    util.init(undefined, inputSize).then(() => {
      updateBackend();
      poseDetectionFrame();
    });
  } else {
    util.init(currentBackend, inputSize).then(() => {
      updateBackend();
    });
  }
}

async function main() {
  let videoSource = await loadVideo();
  detectPoseInRealTime(videoSource);
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

async function predict() {
  isMultiple = guiState.algorithm;
  stats.begin();
  await util.predict(scaleCanvas, ctx, inputSize);
  if (isMultiple == "multi-pose") {
    util.drawOutput(canvas, 'multi', inputSize);
  } else {
    util.drawOutput(canvas, 'single', inputSize);
  }
  stats.end();
}
