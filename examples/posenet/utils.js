/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// mobileNetArchitecture = [layer name, stride]
// conv2d: convolution layer
// separableConv: depthwise convolution layer + pointwise convolution layer
const mobileNet100Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1]
]

const mobileNet75Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1]
]

const mobileNet50Architecture = [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1]
]

class Utils{
  constructor() {
    this.modelArch;
    this.model;
    // single input
    this._version = guiState.model;
    this._outputStride = guiState.outputStride;
    this._minScore = guiState.scoreThreshold;
    this._scaleFactor = guiState.scaleFactor;
    // multiple input
    this._nmsRadius = guiState.multiPoseDetection.nmsRadius;
    this._maxDetection = guiState.multiPoseDetection.maxDetections;
    this._type = "Multiperson";
    this.initialized = false;
    this._cacheMap = new Map();

    this.container = document.getElementById('container');
    this.progressBar = document.getElementById('progressBar');
    this.progressContainer = document.getElementById('progressContainer');
  }
  
  async init(backend, inputSize) {
    this.initialized = false;
    progressContainer.style.display = 'inline';
    progressBar.style = `width: ${0}%`;
    progressBar.innerHTML = `${0}%`;
    let result;
    const ModelArch = new Map([
      [0.5, mobileNet50Architecture],
      [0.75, mobileNet75Architecture],
      [1.0, mobileNet100Architecture],
      [1.01, mobileNet100Architecture],
    ]);
    this.modelArch = ModelArch.get(Number(this._version));
    this.scaleWidth = getValidResolution(this._scaleFactor, inputSize[2], this._outputStride);
    this.scaleHeight = getValidResolution(this._scaleFactor, inputSize[1], this._outputStride);
    this.scaleInputSize = [1, this.scaleWidth, this.scaleHeight, 3];
    if ((this._version == 0.75 || this._version == 0.5) && this._outputStride == 32) {
      this.HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, 16));
    } else {
      this.HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, this._outputStride));
    }
    this.OFFSET_TENSOR_SIZE = this.HEATMAP_TENSOR_SIZE*2;
    this.DISPLACEMENT_FWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;
    this.DISPLACEMENT_BWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;

    this.inputTensor = new Float32Array(this.scaleWidth*this.scaleHeight*3);
    this.heatmapTensor = new Float32Array(this.HEATMAP_TENSOR_SIZE);
    this.offsetTensor = new Float32Array(this.OFFSET_TENSOR_SIZE);
    this.displacementFwd = new Float32Array(this.DISPLACEMENT_FWD_SIZE);
    this.displacementBwd = new Float32Array(this.DISPLACEMENT_BWD_SIZE);
    
    this.model = new PoseNet(this.modelArch, backend, Number(this._version), 
                             Number(this._outputStride), this.scaleInputSize, this._type, this._cacheMap);   
    result = await this.model.createCompiledModel();
    console.log('compilation result: ${result}');
    this.initialized = true;
    if (this.initialized == true) {
      progressContainer.style.display = 'none';
    }
  }

  async predict(scaleCanvas, canvasContextMulti, inputSize) {
    if (!this.initialized) {
      return;
    }
    let imageSize = [this.scaleWidth, this.scaleHeight, 3];
    let scaleData = await this.scaleImage(canvasContextMulti, scaleCanvas, inputSize);
    prepareInputTensor(this.inputTensor, scaleCanvas, this._outputStride, imageSize);
    let start = performance.now();
    let result = await this.model.computeMultiPose(this.inputTensor, this.heatmapTensor, 
                                                   this.offsetTensor, this.displacementFwd, 
                                                   this.displacementBwd);   
    let elapsed = performance.now() - start;
    console.log(`Inference time: ${elapsed.toFixed(2)} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `Inference time: <em style="color:green;font-weight:bloder;">${elapsed.toFixed(2)} </em>ms`; 
  }

  drawOutput(canvas, type, inputSize) {    
    let imageSize = [this.scaleWidth, this.scaleHeight, 3];
    let ctx = canvas.getContext('2d');
    if (type == 'single') {
      let singlePose = decodeSinglepose(this.heatmapTensor, this.offsetTensor, 
                                        toHeatmapsize(imageSize, this._outputStride), 
                                        this._outputStride);
      console.log("single person: ", singlePose);
      singlePose.forEach((pose) => {
        scalePose(pose, inputSize[1]/this.scaleWidth);
        if (pose.score >= this._minScore) {
          drawKeypoints(pose.keypoints, this._minScore, ctx);
          drawSkeleton(pose.keypoints, this._minScore, ctx);
        }
      });
    } else {
      let multiPose = decodeMultiPose(sigmoid(this.heatmapTensor), this.offsetTensor, 
                                      this.displacementFwd, this.displacementBwd, 
                                      this._outputStride, this._maxDetection, this._minScore, 
                                      this._nmsRadius, toHeatmapsize(imageSize, this._outputStride));
      console.log("multiple person: ", multiPose);
      multiPose.forEach((pose) => {
        scalePose(pose, inputSize[1]/this.scaleWidth);
        if (pose.score >= this._minScore) {
          drawKeypoints(pose.keypoints, this._minScore, ctx);
          drawSkeleton(pose.keypoints, this._minScore, ctx);
        }
      });
    }
  }

  scaleImage(canvasContextMulti, scaleCanvas, inputSize) {
    const scale = this.scaleWidth/inputSize[1];
    let pixel = canvasContextMulti.getImageData(0,0, inputSize[1], inputSize[2]);
    scaleCanvas.width = this.scaleWidth;
    scaleCanvas.height = this.scaleHeight;
    scaleCanvas.setAttribute("width", this.scaleWidth);
    scaleCanvas.setAttribute("height", this.scaleHeight);
    let scaleCtx = scaleCanvas.getContext('2d');
    let destImg = scaleCtx.createImageData(this.scaleWidth, this.scaleHeight);
    const promise = new Promise((resolve, reject) => {
      bilinear(pixel, destImg, scale);
      scaleCtx.putImageData(destImg, 0, 0);
      resolve(destImg.data);
    });
    return promise;
  }
}