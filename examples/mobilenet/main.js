const INPUT_TENSOR_SIZE = 224*224*3;
const OUTPUT_TENSOR_SIZE = 1001;

async function loadModelDataFile(url) {
  let response = await fetch(url);
  let arrayBuffer = await response.arrayBuffer();
  let bytes = new Uint8Array(arrayBuffer);
  return bytes;
}

function loadImageToCanvas(url, cavansId) {
  let canvas = document.getElementById(cavansId);
  let ctx = canvas.getContext('2d');
  let img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = function() {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = url;
};

function addFileInputHandler(fileInputId, canvasId) {
  let inputElement = document.getElementById(fileInputId);
  inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
          let imgUrl = URL.createObjectURL(files[0]);
          loadImageToCanvas(imgUrl, canvasId);
      }
  }, false);
}

var tfModel;
var model;
var inputTensor, outputTensor;
function main() {
  inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
  outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
  let predictButton = document.getElementById('predict');
  addFileInputHandler('fileInput', 'canvasInput');
  loadModelDataFile('./model/mobilenet_v1_1.0_224.tflite').then(bytes => {
    let buf = new flatbuffers.ByteBuffer(bytes);
    tfModel = tflite.Model.getRootAsModel(buf);
    printTfLiteModel(tfModel);
    model = new MobileNet(tfModel);
    model.createCompiledModel().then(result => {
      console.log(`compilation result: ${result}`);
      predictButton.removeAttribute('disabled');
    }).catch(e => {
      console.error(e);
    })
  }).catch(e => {
    console.log(e);
  })

  predictButton.addEventListener('click', e => {
    prepareInputTensor(inputTensor, 'canvasInput');
    let start = performance.now();
    model.compute(inputTensor, outputTensor).then(result => {
      let elapsed = performance.now() - start;
      console.log(`Elapsed time: ${elapsed.toFixed(2)} ms`);
      console.log(`Compute result: ${result}`);
      processOutputTensor(outputTensor);
    }).catch(e => {
      console.log(e);
    })
  });
}

function prepareInputTensor(inputTensor, canvasId) {
  let canvas = document.getElementById(canvasId);
  // TODO: implement
}

function processOutputTensor(outputTensor) {
  // TODO: implement
}
