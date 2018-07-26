const util = new Utils();
const canvasSingle = document.getElementById('canvas');
const ctxSingle = canvasSingle.getContext('2d');
const canvasMulti = document.getElementById('canvas_2');
const ctxMulti = canvasMulti.getContext('2d');
async function drawSingleandMulti(){
  let e = document.getElementById("backend");
  let backend = e.options[e.selectedIndex].text;
  switch(backend){
    case "WebGL":
      if (nnPolyfill.supportWebGL2){
        util.init('WebGL2').then(()=>{
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
        await util.init('WASM').then(()=>{
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
        await util.init('WebML').then(()=>{
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
  let minScore = document.getElementById('minpartConfidenceScore').value;
  let nmsRadius = document.getElementById('nmsRadius').value;
  let maxDetection = document.getElementById('maxDetection').value;
  util._minScore = minScore;
  util._nmsRadius = nmsRadius;
  util._maxDetection = maxDetection;
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
    await util.predict(canvasSingle);
    util.drawOutput();
  }else{
    ctxSingle.clearRect(0, 0, canvasSingle.width, canvasSingle.height);
    ctxMulti.clearRect(0, 0, canvasMulti.width, canvasMulti.height);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxSingle);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctxMulti);
    await util.predict(canvasSingle);
    util.drawOutput();
  }
}
