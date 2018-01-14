const INPUT_TENSOR_SIZE = 224*224*3;
const OUTPUT_TENSOR_SIZE = 1001;
const MODEL_FILE = './model/mobilenet_v1_1.0_224.tflite';
const LABELS_FILE = './model/labels.txt';

var tfModel, labels;
var model;
var inputTensor, outputTensor;

function main() {
  inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
  outputTensor = new Float32Array(OUTPUT_TENSOR_SIZE);
  let imageElement = document.getElementById('image');
  let canvasElement = document.getElementById('canvas');
  let canvasContext = canvasElement.getContext('2d');
  let inputElement = document.getElementById('input');
  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
      imageElement.onload = function() {
        canvasContext.drawImage(imageElement, 0, 0,
                                canvasElement.width,
                                canvasElement.height);
      }
    }
  }, false);

  let predictButton = document.getElementById('predict');
  predictButton.addEventListener('click', e => {
    let start = performance.now();
    prepareInputTensor(inputTensor, canvasElement);
    model.compute(inputTensor, outputTensor).then(result => {
      let elapsed = performance.now() - start;
      let classes = getTopClasses(outputTensor, labels);
      console.log(`Elapsed time: ${elapsed.toFixed(2)} ms`);
      console.log(`Compute result: ${result}`);
      console.log(`Classes: `);
      classes.forEach(c => {
        console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      })
    }).catch(e => {
      console.log(e);
    })
  });

  loadModelAndLabels(MODEL_FILE, LABELS_FILE).then(result => {
    labels = result.text.split('\n');
    console.log(`labels: ${labels}`);
    let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
    tfModel = tflite.Model.getRootAsModel(flatBuffer);
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
  });

  fetch(LABELS_FILE).then(response => {
    return response.text();
  }).then(txt => {

  }).catch(e => {
    console.log(e);
  })
}

async function loadModelAndLabels(modelUrl, labelsUrl) {
  let response = await fetch(modelUrl);
  let arrayBuffer = await response.arrayBuffer();
  let bytes = new Uint8Array(arrayBuffer);
  response = await fetch(labelsUrl);
  let text = await response.text();
  return {bytes: bytes, text: text};
}

function prepareInputTensor(tensor, canvas) {
  const width = 224;
  const height = 224;
  const channels = 3;
  const imageChannels = 4; // RGBA
  const mean = 127.5;
  const std = 127.5;
  if (canvas.width !== width || canvas.height !== height) {
    throw new Error(`canvas.width(${canvas.width}) or canvas.height(${canvas.height}) is not 224`);
  }
  let context = canvas.getContext('2d');
  let pixels = context.getImageData(0, 0, width, height).data;
  // NHWC layout
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      for (let c = 0; c < channels; ++c) {
        let value = pixels[y*width*imageChannels + x*imageChannels + c];
        tensor[y*width*channels + x*channels + c] = (value - mean)/std;
      }
    }
  }
}

function getTopClasses(tensor, labels, k = 5) {
  let probs = Array.from(tensor);
  let indexes = probs.map((prob, index) => [prob, index]);
  let sorted = indexes.sort((a, b) => {
    if (a[0] === b[0]) {return 0;}
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();
  let classes = [];
  for (let i = 0; i < k; ++i) {
    let prob = sorted[i][0];
    let index = sorted[i][1];
    let c = {
      label: labels[index],
      prob: (prob * 100).toFixed(2)
    }
    classes.push(c);
  }
  return classes;
}
