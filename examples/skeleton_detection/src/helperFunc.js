/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

function getInput(inputElement) {
  console.log(inputElement)
  let reader = new FileReader();
  console.log(reader)
  const promise = new Promise((resolve, reject) => {
    reader.onload = function(e){
      resolve(e.target.result);
    }
    reader.readAsDataURL(inputElement);
  });
  return promise;
}

function getURL(version) {
  let address;
  const urlBase = 'https://storage.googleapis.com/tfjs-models/weights/posenet/';
  // const urlBase = '../skeleton_detection/model/';
  switch (version) {
    case 1.01:
      address = urlBase + 'mobilenet_v1_101/';
      break;
    case 1.0:
      address = urlBase + 'mobilenet_v1_100/';
      break;
    case 0.75:
      address = urlBase + 'mobilenet_v1_075/'; 
      break;
    case 0.5:
      address = urlBase + 'mobilenet_v1_050/';
      break;
    default:
      throw new Error('It must be 1.01, 1.0, 0.75 or 0.5');
  }
  return address;
}

// Obtain weights data and bias data
async function fetchDataByUrl(url, binary) {
  return new Promise(function(resolve, reject) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    if (binary) {
      xhr.responseType = 'arraybuffer';
    }
    xhr.onload = function(ev) {
      if (xhr.readyState == 4) {
        if (xhr.status == 200) {
          resolve(xhr.response);
        } else {
          reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
        }
      }
    };
    xhr.send();
  });
}

async function getDimensionData(layername, version, blockId, manifest, cacheMap) {
  if (layername == 'conv2d') {
    let layerWeights = 'MobilenetV1/Conv2d_'+String(blockId)+'/weights';
    let layerBias = 'MobilenetV1/Conv2d_'+String(blockId)+'/biases';
    let shapeWeights = manifest[layerWeights]['shape'];
    let shapeBias = manifest[layerBias]['shape'];
    let filenameWeights = manifest[layerWeights]['filename'];
    let filenameBias = manifest[layerBias]['filename'];
    let addressWeights = getURL(version)+filenameWeights;
    let addressBias = getURL(version)+filenameBias;
    let weights = await loadCache(addressWeights, cacheMap);
    let bias = await loadCache(addressBias, cacheMap);
    weights = new Float32Array(weights);
    bias = new Float32Array(bias);
    return {shapeWeights: shapeWeights, weights: weights, shapeBias: shapeBias, bias: bias};
  }
  else if (layername == 'separableConv') {
    let layerDepthWeights = 'MobilenetV1/Conv2d_'+String(blockId)+'_depthwise/depthwise_weights';
    let layerPointWeights = 'MobilenetV1/Conv2d_'+String(blockId)+'_pointwise/weights';
    let layerDepthBias = 'MobilenetV1/Conv2d_'+String(blockId)+'_depthwise/biases';
    let layerPointBias = 'MobilenetV1/Conv2d_'+String(blockId)+'_pointwise/biases';
    let shapeWeights = [];
    let shapeBias = [];
    let weights = [];
    let bias = [];
    shapeWeights.push(manifest[layerDepthWeights]['shape']);
    shapeWeights.push(manifest[layerPointWeights]['shape']);
    shapeBias.push(manifest[layerDepthBias]['shape']);
    shapeBias.push(manifest[layerPointBias]['shape']);
    let fileDepthWeights = manifest[layerDepthWeights]['filename'];
    let filePointWeights = manifest[layerPointWeights]['filename'];
    let fileDepthBias = manifest[layerDepthBias]['filename'];
    let filePointBias = manifest[layerPointBias]['filename'];
    let depthWeights = await loadCache(getURL(version)+fileDepthWeights, cacheMap);
    let pointWeights = await loadCache(getURL(version)+filePointWeights, cacheMap);
    let depthBias = await loadCache(getURL(version)+fileDepthBias, cacheMap);
    let pointBias = await loadCache(getURL(version)+filePointBias, cacheMap);
    weights.push(new Float32Array(depthWeights));
    weights.push(new Float32Array(pointWeights));
    bias.push(new Float32Array(depthBias));
    bias.push(new Float32Array(pointBias));
    return {shapeWeights: shapeWeights, weights: weights, shapeBias: shapeBias, bias: bias};
  } else {
    let shapeWeights;
    let shapeBias;
    shapeWeights = manifest['MobilenetV1/'+layername+'_2/weights']['shape'];
    shapeBias = manifest['MobilenetV1/'+layername+'_2/biases']['shape'];
    let weights = await loadCache(getURL(version)+manifest['MobilenetV1/'+layername+'_2/weights']['filename'], cacheMap);
    let bias = await loadCache(getURL(version)+manifest['MobilenetV1/'+layername+'_2/biases']['filename'], cacheMap);
    weights = new Float32Array(weights);
    bias = new Float32Array(bias);
    return {shapeWeights: shapeWeights, weights: weights, shapeBias: shapeBias, bias: bias};
  }
}

async function loadCache(address, cacheMap) {
  let results;
  if (cacheMap.get(address) == undefined) {
    results = await fetchDataByUrl(address, true);
    cacheMap.set(address, results);
  } else {
    results = cacheMap.get(address); 
  }
  return results;
}

// obtain desired size output
function toOutputStridedLayers(convolutionDefinition, outputStride) {
  let currentStride = 1;
  let rate = 1;
  return convolutionDefinition.map(function (_a, blockId) {
    let convType = _a[0], stride = _a[1];
    let layerStride, layerRate;
    if (currentStride === outputStride) {
      layerStride = 1;
      layerRate = rate;
      rate *= stride;
    } else {
      layerStride = stride;
      layerRate = 1;
      currentStride *= stride;
    }
    return {
      blockId: blockId, convType: convType, stride: layerStride, rate: layerRate,
      outputStride: currentStride
    };
  });
}

// weights dimension: HWCN -> NHWC
function reshape(dimension) {
  return [dimension[3], dimension[0], dimension[1], dimension[2]];
}

// HWCN -> NHWC
function transposeWeights(weights, dimension) {
  let product = dimension.reduce(function(a,b){return a*b});
  let newWeights = new Float32Array(product);
  let [H, W, C, N] = dimension;
  for (let h =0; h<H; h++) {
    for (let w =0; w<W; w++) {
      for (let c = 0; c<C; c++) {
        for (let n =0; n<N; n++) {
          newWeights[c + w*C + h*W*C + n*H*W*C] = 
              weights[n + c*N + w*C*N + h*W*C*N];
        }
      }
    }
  }
  return newWeights;
}

function valideResolution(inputDimension, outputStride) {
  let width = inputDimension[1];
  let height = inputDimension[2];
  if ((width-1) % outputStride != 0) {
    throw new Error('invalid resolution');
  }
}

function prepareInputTensor(tensor, canvas, outputStride, imgDimension) {
  const width = imgDimension[1];
  const height = imgDimension[2];
  const channels = imgDimension[3];
  const imageChannels = 4;
  const mean = 127.5;
  const std = 127.5;
  valideResolution(imgDimension, outputStride);
  let context = canvas.getContext('2d');
  let pixels = context.getImageData(0, 0, width, height).data;
  for (let y = 0; y < height; ++y) {
    for (let x = 0; x < width; ++x) {
      for (let c = 0; c < channels; ++c) {
        let value = pixels[y*width*imageChannels + x*imageChannels + c];
        tensor[y*width*channels + x*channels + c] = (value - mean)/std;
      }
    }
  }   
}

function sigmoid(heatmap) {
  let heatmapScore = [];
  for (let i in heatmap) {
    heatmapScore.push(1/(1+Math.pow(Math.E, -heatmap[i])));
  }
  return heatmapScore;
}

function argMax(array) {
  let max = 0;
  for (let i in array) {
    if (array[i] > array[max]) {
      max = i;
    }
  }
  return max;
}

function getKeypointIndex(array, dimension) {
  let newArray = [];
  let index = [];
  let confidenceScore = [];
  for (let i = 0; i< 17; i++) {
    newArray.push(new Array(dimension[0]*dimension[1]));
  }
  for (let i in array) {
    newArray[i%17][Math.floor(i/17)] = array[i];
  }
  for (let j in newArray) {
    index.push(argMax(newArray[j]));
    confidenceScore.push(newArray[j][argMax(newArray[j])]);
  }
  return [index, confidenceScore];
}

function convertPosition(index, dimension) {
  let height = dimension[0];
  let width = dimension[1];
  let x = index%height;
  let y = (Math.floor(index/height))%width;
  return [y, x];
}

function convertIndextoCoor(index, dimension) {
  [height, width, channel] = dimension;
  let z = index%channel;
  let x = Math.floor(index/channel)%width;
  let y = Math.floor(index/(width*channel));
  return [y, x, z];
}

function convertCoortoIndex(x, y, z, dimension) {
  let [height, width, channel] = dimension;
  let index = Number(z)+Number(x*channel)+Number(y*width*channel);
  return index;
}

function decodeSinglepose(heatmap, offset, dimension, outputStride) {
  let [index, confidenceScore] = getKeypointIndex(heatmap, dimension);
  let finalRes = [];
  let totalScore = 0; 
  let poses = [];
  for (let i in index) {
    let heatmapY = convertPosition(Number(index[i]), dimension)[0];
    let heatmapX = convertPosition(Number(index[i]), dimension)[1];
    let offsetY = offset[Number(i)+Number(heatmapX)*34+Number(heatmapY*34*dimension[1])];
    let offsetX = offset[Number(i)+17+Number(heatmapX)*34+Number(heatmapY*34*dimension[1])];
    let finalPos = [heatmapY*outputStride+offsetY, heatmapX*outputStride+offsetX];
    totalScore += confidenceScore[i];
    finalRes.push({position: {y: finalPos[0], x: finalPos[1]}, 
                   part: partNames[i], score: confidenceScore[i]});
  }
  poses.push({keypoints: finalRes, score: totalScore/17});
  return poses;
}

function squaredDistance(y1, x1, y2, x2) {
  let dy = y2 - y1;
  let dx = x2 - x1;
  return dy * dy + dx * dx;
}

function withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, _a, keypointId) {
  let x = _a.x, y = _a.y;
  return poses.some(function (_a) {
    let keypoints = _a.keypoints;
    let correspondingKeypoint = keypoints[keypointId].position;
    return squaredDistance(y, x, correspondingKeypoint.y, correspondingKeypoint.x) <= squaredNmsRadius;
  });
}

function scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores, dimension) {
  let height = dimension[0], width = dimension[1];
  let localMaximum = true;
  let yStart = Math.max(heatmapY - localMaximumRadius, 0);
  let yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
  for (let yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
    let xStart = Math.max(heatmapX - localMaximumRadius, 0);
    let xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
    for (let xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
      let index = convertCoortoIndex(xCurrent, yCurrent, keypointId, dimension);
      if (scores[index] > score) {
        localMaximum = false;
        break;
      }
    }
    if (!localMaximum) {
      break;
    }
  }
  return localMaximum;
}

function toHeatmapsize(dimension, outputStride) {
  let heatmapSize;
  if (dimension.length == 3) {
    heatmapSize = [(dimension[0]-1)/outputStride+1, (dimension[1]-1)/outputStride+1, 17];
  }
  if (dimension.length == 4) {
    heatmapSize = [(dimension[1]-1)/outputStride+1, (dimension[2]-1)/outputStride+1, 17]
  }
  return heatmapSize;
}

function product(array) {
  return array.reduce(function(a,b){return a*b;});
}

function buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores, dimension) {
  const height = dimension[0];
  const width = dimension[1];
  const numKeypoints = dimension[2];
  let queue = new MaxHeap(height*width*numKeypoints, function(_a){
    let score = _a.score;
    return score;
  });
  for (let heatmapY = 0; heatmapY < height; ++heatmapY) {
    for (let heatmapX = 0; heatmapX < width; ++heatmapX) {
      for (let keypointId = 0; keypointId < numKeypoints; ++keypointId) {
        let index = convertCoortoIndex(heatmapX, heatmapY, keypointId, dimension);
        let score = scores[index];
        if (score < scoreThreshold) {
          continue;
        }
        if (scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores, dimension)) {
          queue.enqueue({ score: score, part: { heatmapY: heatmapY, heatmapX: heatmapX, id: keypointId } });
        }
      }
    }
  }
  return queue;
}

function getImageCoords(part, outputStride, offsets, dimension) {
  let dimensionOffset = [];
  dimensionOffset.push(dimension[0]);
  dimensionOffset.push(dimension[1]);
  dimensionOffset.push(34);
  let heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
  let indexY = convertCoortoIndex(heatmapX, heatmapY, keypoint, dimensionOffset);
  let indexX = indexY+17;
  return {
    x: part.heatmapX * outputStride + offsets[indexX],
    y: part.heatmapY * outputStride + offsets[indexY]
  };
}

function getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints) {
  let notOverlappedKeypointScores = instanceKeypoints.reduce(function (result, _a, keypointId) {
    let position = _a.position, score = _a.score;
    if (!withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, position, keypointId)) {
      result += score;
    }
    return result;
  }, 0.0);
  return notOverlappedKeypointScores /= instanceKeypoints.length;
}

function decodeMultiPose(heatmap, offsets, displacementFwd, displacementBwd, outputStride, 
                         maxPoseDectection, scoreThreshold, nmsRadius, dimension) {
  let poses = [];
  let queue = buildPartWithScoreQueue(scoreThreshold, 1, heatmap, dimension);
  let _root, keypoints, score;
  const squaredNmsRadius = nmsRadius * nmsRadius;
  let index = 0;
  while (poses.length < maxPoseDectection && !queue.empty()) {
    _root = queue.dequeue();
    rootImgCoord = getImageCoords(_root.part, outputStride, offsets, dimension);
    if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImgCoord, _root.part.id)) {
      continue;
    }
    keypoints = decodePose(_root, heatmap, offsets, outputStride, displacementFwd, displacementBwd, dimension);
    score = getInstanceScore(poses, squaredNmsRadius, keypoints);
    poses.push({keypoints: keypoints, score: score});
  }
  return poses;
}   

function convert4D(n, h, w, c, dimension) {
  let index = Number(c)+Number(w*dimension[3])+Number(h*dimension[3]*dimension[2])+
              Number(n*dimension[1]*dimension[2]*dimension[3]);
  return index;
}

function dilationWeights(weights, dimension, rate) {
  let dilationW = dimension[2]*rate-rate+1;
  let dilationH = dimension[1]*rate-rate+1;
  let dilationWeights = new Float32Array(dimension[0]*dilationW*dilationH*dimension[3]);
  dilationWeights.fill(0);
  let dimensionDilation = [dimension[0], dilationH, dilationW, dimension[3]];
  for (let h = 0; h < dilationH; h += rate) {
    for (let w = 0; w < dilationW; w += rate) {
      for (let c = 0; c < dimension[3]; c++) {
        let indexDilation = convert4D(0, h, w, c, dimensionDilation);
        let indexOrigin = convert4D(0, h/rate, w/rate, c, dimension);
        dilationWeights[indexDilation] = weights[indexOrigin];
      }
    }
  }
  return {dimension: dimensionDilation, dilationWeights: dilationWeights}
}

function getValidResolution(imageScaleFactor, inputDimension, outputStride) {
  let evenResolution = inputDimension * imageScaleFactor - 1;
  return evenResolution - (evenResolution % outputStride) + 1;
}

function ivect(ix, iy, w) {
  // byte array, r,g,b,a
  return((ix + w * iy) * 4);
}

function bilinear(srcImg, destImg, scale) {
  function inner(f00, f10, f01, f11, x, y) {
    var un_x = 1.0 - x; var un_y = 1.0 - y;
    return (f00 * un_x * un_y + f10 * x * un_y + f01 * un_x * y + f11 * x * y);
  }
  var i, j;
  var iyv, iy0, iy1, ixv, ix0, ix1;
  var idxD, idxS00, idxS10, idxS01, idxS11;
  var dx, dy;
  var r, g, b, a;
  for (i = 0; i < destImg.height; ++i) {
    iyv = i / scale;
    iy0 = Math.floor(iyv);
    // Math.ceil can go over bounds
    iy1 = ( Math.ceil(iyv) > (srcImg.height-1) ? (srcImg.height-1) : Math.ceil(iyv) );
    for (j = 0; j < destImg.width; ++j) {
      ixv = j / scale;
      ix0 = Math.floor(ixv);
    
      // Math.ceil can go over bounds
      ix1 = ( Math.ceil(ixv) > (srcImg.width-1) ? (srcImg.width-1) : Math.ceil(ixv) );
      idxD = ivect(j, i, destImg.width);
    
      // matrix to vector indices
      idxS00 = ivect(ix0, iy0, srcImg.width);
      idxS10 = ivect(ix1, iy0, srcImg.width);
      idxS01 = ivect(ix0, iy1, srcImg.width);
      idxS11 = ivect(ix1, iy1, srcImg.width);
    
      // overall coordinates to unit square
      dx = ixv - ix0; dy = iyv - iy0;
    
      r = inner(srcImg.data[idxS00], srcImg.data[idxS10],
                srcImg.data[idxS01], srcImg.data[idxS11], dx, dy);
      destImg.data[idxD] = r;

      g = inner(srcImg.data[idxS00+1], srcImg.data[idxS10+1],
                srcImg.data[idxS01+1], srcImg.data[idxS11+1], dx, dy);
      destImg.data[idxD+1] = g;

      b = inner(srcImg.data[idxS00+2], srcImg.data[idxS10+2],
                srcImg.data[idxS01+2], srcImg.data[idxS11+2], dx, dy);
      destImg.data[idxD+2] = b;

      a = inner(srcImg.data[idxS00+3], srcImg.data[idxS10+3],
                srcImg.data[idxS01+3], srcImg.data[idxS11+3], dx, dy);
      destImg.data[idxD+3] = a;
    }
  }
}

function getBoundingBox(keypoints) {
  const NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY; 
  const POSITIVE_INFINITY = Number.POSITIVE_INFINITY;
  return keypoints.reduce(function (_a, _b) {
    var maxX = _a.maxX, maxY = _a.maxY, minX = _a.minX, minY = _a.minY;
    var _c = _b.position, x = _c.x, y = _c.y;
    return {
      maxX: Math.max(maxX, x),
      maxY: Math.max(maxY, y),
      minX: Math.min(minX, x),
      minY: Math.min(minY, y)
    };
  }, {
      maxX: NEGATIVE_INFINITY,
      maxY: NEGATIVE_INFINITY,
      minX: POSITIVE_INFINITY,
      minY: POSITIVE_INFINITY
  });
}
