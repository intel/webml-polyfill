const util = new Utils();
const canvasSingle = document.getElementById('canvasSingle');
const ctxSingle = canvasSingle.getContext('2d');
const canvasMulti = document.getElementById('canvasMulti');
const ctxMulti = canvasMulti.getContext('2d');
const scaleCanvas = document.getElementById('scaleImage');
const scaleCtx = scaleCanvas.getContext('2d');
const inputSize = [1, 513, 513, 3];
const backend = document.getElementById('backend');
const wasm = document.getElementById('wasm');
const webgl = document.getElementById('webgl');
const webml = document.getElementById('webml');
let currentBackend = '';

function main() {
  function updateBackend() {
    currentBackend = util.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend) {
    console.log(newBackend);
    if (currentBackend === newBackend) {
      return;
    }
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      util.init(newBackend, inputSize).then(() => {
        updateBackend();
        drawResult();
      });
    }, 10);
  }

  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function(e) {
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
      drawResult();
      button.setAttribute('class', 'btn btn-primary');
      image.removeAttribute('disabled');
    });
  } else {
    util.init(currentBackend, inputSize).then(() => {
      updateBackend();
      drawResult();
    });
  }
}

model.onChange((model) => {
  guiState.model = model;
  util._version = guiState.model;
  main();
});
outputStride.onChange((outputStride) => {
  guiState.outputStride = outputStride;
  util._outputStride = guiState.outputStride;
  main();
});
scaleFactor.onChange((scaleFactor) => {
  guiState.scaleFactor = scaleFactor;
  util._scaleFactor = guiState.scaleFactor;
  main();
});
scoreThreshold.onChange((scoreThreshold) => {
  guiState.scoreThreshold = scoreThreshold;
  util._minScore = guiState.scoreThreshold;
  drawResult();
});
nmsRadius.onChange((nmsRadius) => {
  guiState.multiPoseDetection.nmsRadius = nmsRadius;
  util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
  drawResult();
});
maxDetections.onChange((maxDetections) => {
  guiState.multiPoseDetection.maxDetections = maxDetections;
  util._maxDetection = guiState.multiPoseDetection.maxDetections;
  drawResult();
});


async function drawResult() {
  let _inputElement = document.getElementById('image').files[0];
  ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
  ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
  if (_inputElement != undefined) {
    let x = await getInput(_inputElement);
    await loadImage(x, ctxSingle);
    await loadImage(x, ctxMulti);
    await util.predict(scaleCanvas, ctxMulti, inputSize);
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  } else {
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxSingle);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxMulti);
    await util.predict(scaleCanvas, ctxMulti, inputSize);
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  }
}
