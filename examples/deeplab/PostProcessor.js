onmessage = function (e) {
  let id = e.data[0];
  let fn = e.data[1];
  let args = e.data[2];
  let ret = eval(fn)(...args);

  // only support transferring buffer property, spcifically ArrayBuffer
  let trans = ret.filter(o => o.hasOwnProperty('transfer')).map(o => o.buffer);
  postMessage([id, fn, ret], trans);
};

function _move(obj) {
  obj.transfer = true; // tagged for transfer
  return obj;
}


let predictions = null;
let segMap = null;

function colorizeAndPredictLabels(newSegMap) {
  let start = performance.now();

  // colorize segmentation map
  segMap = newSegMap;
  let rawSegMapData = segMap.data;
  let numClasses = segMap.outputShape[2];
  predictions = argmax(rawSegMapData, numClasses);
  let colorSegMap = colorizeSegMap(predictions, segMap.outputShape);

  // generate label map. { labelName: [ labelName, rgbTuple ] }
  let uniqueLabels = new Set(predictions);
  let labelMap = {};
  for (let labelId of uniqueLabels) {
    let labelName = segMap.labels[labelId];
    let rgbTuple = palette[labelId].slice(0, 3);
    labelMap[labelId] = [labelName, rgbTuple];
  }

  console.log(`[Worker] Draw time: ${(performance.now() - start).toFixed(2)} ms`);
  return [_move(colorSegMap), labelMap];
}

function getHoverLabelId(hoverPos) {
  let outputW = segMap.outputShape[0];
  let hoverLabelId = predictions[hoverPos.x + hoverPos.y * outputW];
  return [hoverLabelId];
}

const palette = [
  [45, 52, 54, 255],
  [85, 239, 196, 255],
  [129, 236, 236, 255],
  [116, 185, 255, 255],
  [162, 155, 254, 255],
  [223, 230, 233, 255],
  [0, 184, 148, 255],
  [0, 206, 201, 255],
  [9, 132, 227, 255],
  [39, 60, 117, 255],
  [108, 92, 231, 255],
  [178, 190, 195, 255],
  [255, 234, 167, 255],
  [250, 177, 160, 255],
  [255, 118, 117, 255],
  [253, 121, 168, 255],
  [99, 110, 114, 255],
  [253, 203, 110, 255],
  [225, 112, 85, 255],
  [214, 48, 49, 255],
  [232, 67, 147, 255],
];

function argmax(array, span) {
  const len = array.length / span;
  const result = new Array(len);
  for (let i = 0; i < len; i++) {
    let maxVal = Number.MIN_SAFE_INTEGER;
    let maxIdx = 0;
    for (let j = 0; j < span; j++) {
      if (array[i * span + j] > maxVal) {
        maxVal = array[i * span + j];
        maxIdx = j;
      }
    }
    result[i] = maxIdx;
  }
  return result;
}

function colorizeSegMap(predictions, outputShape) {
  const outputW = outputShape[0];
  const outputH = outputShape[1];
  const imageChannels = 4;
  const colorSegMap = new Uint8ClampedArray(outputW * outputH * imageChannels);
  for (let i = 0, j = 0; i < predictions.length; i++, j += imageChannels) {
    let color = palette[predictions[i]];
    colorSegMap[j] = color[0];
    colorSegMap[j + 1] = color[1];
    colorSegMap[j + 2] = color[2];
    colorSegMap[j + 3] = color[3];
  }
  return colorSegMap;
}