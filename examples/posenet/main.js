guiState.scaleFactor = 0.75;

const util = new Utils();
const canvasSingle = document.getElementById('canvasSingle');
const ctxSingle = canvasSingle.getContext('2d');
const canvasMulti = document.getElementById('canvasMulti');
const ctxMulti = canvasMulti.getContext('2d');
const scaleCanvas = document.getElementById('scaleImage');
const scaleCtx = scaleCanvas.getContext('2d');
const inputWidth = 513;
const inputHeight = 513;
const inputSize = [1, inputWidth, inputHeight, 3];
const backend = document.getElementById('backend');
const wasm = document.getElementById('wasm');
const webgl = document.getElementById('webgl');
const webml = document.getElementById('webml');
const inputImage = document.getElementById('image');
const selectPrefer = document.getElementById('selectPrefer');
let currentBackend = '';
let currentPrefer = '';

function checkPreferParam() {
  if (currentOS === 'Mac OS') {
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
  div.innerHTML = `<strong>Currently ${backend} backend doesn't support PoseNet Model.</strong>`;
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

function updateBackend() {
  if (getUrlParams('api_info') === 'true') {
    backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI(currentPrefer) : currentBackend;
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
  util.deleteAll();
  backend.innerHTML = 'Setting...';
  setTimeout(() => {
    util.init(newBackend, currentPrefer, inputSize).then(() => {
      currentBackend = newBackend;
      updatePrefer();
      updateBackend();
      drawResult();
    }).catch((e) => {
      console.warn(`Failed to init ${util.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(util.model._backend);
      changeBackend('WASM');
      updatePrefer();
      backend.innerHTML = 'WASM';
    });
  }, 10);
}

function changePrefer(newPrefer, force) {
  if (currentPrefer === newPrefer && !force) {
    return;
  }
  util.deleteAll();
  removeAlertElement();
  selectPrefer.innerHTML = 'Setting...';
  setTimeout(() => {
    util.init(currentBackend, newPrefer, inputSize).then(() => {
      currentPrefer = newPrefer;
      updatePrefer();
      updateBackend();
      drawResult();
    }).catch((e) => {
      let currentBackend = 'WebML/' + getNativeAPI(currentPrefer);
      let nextBackend = 'WebML/' + getNativeAPI(newPrefer);
      console.warn(`Failed to change backend ${nextBackend}, switch back to ${currentBackend}`);
      console.error(e);
      changePrefer(currentPrefer, true);
      showAlert(nextBackend);
      updatePrefer();
      updateBackend();
    });
  }, 10);
}

function updatePrefer() {
  selectPrefer.innerHTML = preferMap[currentPrefer];
}

function main() {
  checkPreferParam();

  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function(e) {
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
  if (currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let sustained = $('<button class="dropdown-item"/>')
      .text('SUSTAINED_SPEED')
      .click(_ => changePrefer('sustained'));
    $('.preference').append(sustained);
    if (currentOS === 'Android') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item"/>')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Windows' || currentOS === 'Linux') {
      let fast = $('<button class="dropdown-item" disabled />')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Mac OS') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    }
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  initModel(true);
}

function initModel(first = false) {
  if (!first && !util.initialized) {
    console.warn('not initialized');
    return;
  }
  util.init(currentBackend, currentPrefer, inputSize).then(() => {
    updateBackend();
    updatePrefer()
    drawResult();
    button.setAttribute('class', 'btn btn-primary');
    image.removeAttribute('disabled');
  }).catch((e) => {
    console.warn(`Failed to init ${util.model._backend}, try to use WASM`);
    console.error(e);
    showAlert(util.model._backend);
    changeBackend('WASM');
  });
}

inputImage.addEventListener('change', () => {
  drawResult();
})

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

scoreThreshold.onChange((scoreThreshold) => {
  guiState.scoreThreshold = parseFloat(scoreThreshold);
  util._minScore = guiState.scoreThreshold;
  drawResult(false, false);
});

nmsRadius.onChange((nmsRadius) => {
  guiState.multiPoseDetection.nmsRadius = parseInt(nmsRadius);
  util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
  drawResult(false, true);
});

maxDetections.onChange((maxDetections) => {
  guiState.multiPoseDetection.maxDetections = parseInt(maxDetections);
  util._maxDetection = guiState.multiPoseDetection.maxDetections;
  drawResult(false, true);
});

showPose.onChange((showPose) => {
  guiState.showPose = showPose;
  drawResult(false, false);
});

showBoundingBox.onChange((showBoundingBox) => {
  guiState.showBoundingBox = showBoundingBox;
  drawResult(false, false);
});

function drawImage(image, canvas, w, h) {
  const ctx = canvas.getContext('2d');
  canvas.width = w;
  canvas.height = h;
  canvas.setAttribute("width", w);
  canvas.setAttribute("height", h);
  ctx.save();
  ctx.drawImage(image, 0, 0, w, h);
  ctx.restore();
}

function loadImage(imagePath, canvas) {
  const ctx = canvas.getContext('2d');
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      canvas.width = inputWidth;
      canvas.height = inputHeight;
      canvas.setAttribute("width", inputWidth);
      canvas.setAttribute("height", inputHeight);
      ctx.drawImage(image, 0, 0, inputWidth, inputHeight);
      resolve(image);
    };
  });
  image.src = imagePath;
  return promise;
}

let singlePose, multiPoses;
async function drawResult(predict = true, decode = true) {
  if (!util.initialized) {
    console.warn('not initialized');
    return;
  }
  let _inputElement = document.getElementById('image').files[0];
  let imageUrl;
  if (_inputElement != undefined) {
    imageUrl = await getInput(_inputElement);
  } else {
    imageUrl = 'https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg';
  }
  await loadImage(imageUrl, canvasSingle);
  let image = await loadImage(imageUrl, canvasMulti);
  drawImage(image, scaleCanvas, util.scaleWidth, util.scaleHeight);
  let predictTime = 0, decodeTime = 0;
  if (predict) {
    await util.predict(scaleCanvas, 'single');
    const start = performance.now();
    await util.predict(scaleCanvas, 'multi');
    predictTime = performance.now() - start;
  }
  if (decode) {
    singlePose = util.decodePose('single');
    const start = performance.now();
    multiPoses = util.decodePose('multi');
    decodeTime = performance.now() - start;
  }
  if (predict && decode) {
    const elapsed = predictTime + decodeTime;
    const inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `Inference time: <em style="color:green;font-weight:bloder;">${elapsed.toFixed(2)} </em>ms`;
  }
  util.drawPoses(canvasSingle, singlePose);
  util.drawPoses(canvasMulti, multiPoses);
}