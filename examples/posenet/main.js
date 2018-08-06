const util = new Utils();
const canvasSingle = document.getElementById('canvas');
const ctxSingle = canvasSingle.getContext('2d');
const canvasMulti = document.getElementById('canvas_2');
const ctxMulti = canvasMulti.getContext('2d');
const scaleCanvas = document.getElementById('scaleImage');
const scaleCtx = scaleCanvas.getContext('2d');
const inputSize = [1, 513, 513, 3];
async function drawSingleandMulti(){
  let e = document.getElementById("backend");
  let backend = e.options[e.selectedIndex].text;
  switch(backend){
    case "WebGL":
      if (nnPolyfill.supportWebGL2){
        util.init('WebGL2', inputSize).then(()=>{
          drawResult();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('main').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WebGL");
      }
      break;
    case "WASM":
      if (nnPolyfill.supportWasm){
        await util.init('WASM', inputSize).then(()=>{
          drawResult();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('main').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WASM");
      }
      break;
    case "WebML":
      if(nnNative){
        await util.init('WebML', inputSize).then(()=>{
          drawResult();
          document.getElementById('loading').style.display = 'none';
          document.getElementById('main').style.display = 'block';
        });
      }
      else{
        throw new Error("Do not support WebML");
      }
      break;
    default:
      break;
  }
}

async function updateParameter(){
  util._minScore = guiState.scoreThreshold;
  util._nmsRadius = guiState.multiPoseDetection.nmsRadius;
  util._maxDetection = guiState.multiPoseDetection.maxDetections;
  drawResult();
}

async function drawResult(){
  let _inputElement = document.getElementById('image').files[0];
  if(_inputElement!=undefined){
    ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
    ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
    let x = await getInput(_inputElement);
    await loadImage(x, ctxSingle);
    await loadImage(x, ctxMulti);
    await util.predict(scaleCanvas, ctxMulti, inputSize);
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  }else{
    ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
    ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxSingle);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxMulti);
    await util.predict(scaleCanvas, ctxMulti, inputSize);
    util.drawOutput(canvasMulti, 'multi', inputSize);
    util.drawOutput(canvasSingle, 'single', inputSize);
  }
}
