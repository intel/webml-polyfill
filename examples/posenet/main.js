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
const inputImage = document.getElementById('image');
let currentBackend = '';
let predictStatus = true;

function main() {
  predictStatus = true;
  function updateBackend() {
    currentBackend = util.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getActuralNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend) {
    console.log(newBackend);
    predictStatus = true;
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

inputImage.addEventListener('change', () => {
  predictStatus = true;
  drawResult();
})
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
showPose.onChange((showPose) => {
  guiState.showPose = showPose;
  drawResult();
});
showBoundingBox.onChange((showBoundingBox) => {
  guiState.showBoundingBox = showBoundingBox;
  drawResult();
});


async function drawResult() {
  let _inputElement = document.getElementById('image').files[0];
  if (_inputElement != undefined) {
    let x = await getInput(_inputElement);
    await loadImage(x, ctxSingle);
    await loadImage(x, ctxMulti);
    if (predictStatus) {
      await util.predict(scaleCanvas, ctxMulti, inputSize, 'multi');
      predictStatus = false;
    }
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  } else {
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxSingle);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxMulti);    
    if (predictStatus) {
      await util.predict(scaleCanvas, ctxMulti, inputSize, 'multi');
      predictStatus = false;
    }
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  }
}
