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
const inputSize = [1, 513, 513, 3];

class Utils{
  constructor(){
    this.tfmodel;
    this.model;
    // single input
    this._version = guiState.model;
    this._outputStride = guiState.outputStride;
    this._minScore = guiState.scoreThreshold;
    this._scaleFactor = guiState.scaleFactor;
    // multiple input
    this._nmsRadius = guiState.multiPoseDetection.nmsRadius;
    this._maxDetection = guiState.multiPoseDetection.maxDetections;
    
    this.canvasElementSingle = document.getElementById('canvas');
    this.canvasContextSingle = this.canvasElementSingle.getContext('2d');
    this.canvasElementMulti = document.getElementById('canvas_2');
    this.canvasContextMulti = this.canvasElementMulti.getContext('2d');
    this.scaleCanvas = document.getElementById('scaleImage');
    this.scaleCtx = this.scaleCanvas.getContext('2d');
    this._type = "Multiperson";
    this.initialized = false;
  }

  async init(backend){
    this.initialized = false;
    let result;
    if(this._minScore<0 | this._minScore>1){
      alert("Minimal Part Confidence Score must be in range (0,1).");
      return;
    }
    if(this._outputStride!=8 & this._outputStride!=16 & this._outputStride!=32){
      alert("OutputSride must be 8, 16 or 32");
      return;
    }
    if(!this.tfmodel){
      const ModelArch = new Map([
        [0.5, mobileNet50Architecture],
        [0.75, mobileNet75Architecture],
        [1.0, mobileNet100Architecture],
        [1.01, mobileNet100Architecture],
      ]);
      this.tfmodel = ModelArch.get(Number(this._version));
    }   

    this.scaleWidth = getValidResolution(this._scaleFactor, inputSize[2], this._outputStride);
    this.scaleHeight = getValidResolution(this._scaleFactor, inputSize[1], this._outputStride);
    this.scaleInputSize = [1, this.scaleWidth, this.scaleHeight, 3];
    this.HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.scaleInputSize, this._outputStride));
    this.OFFSET_TENSOR_SIZE = this.HEATMAP_TENSOR_SIZE*2;
    this.DISPLACEMENT_FWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;
    this.DISPLACEMENT_BWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;

    this.inputTensor = new Float32Array(this.scaleWidth*this.scaleHeight*3);
    this.heatmapTensor = new Float32Array(this.HEATMAP_TENSOR_SIZE);
    this.offsetTensor = new Float32Array(this.OFFSET_TENSOR_SIZE);
    this.displacement_fwd = new Float32Array(this.DISPLACEMENT_FWD_SIZE);
    this.displacement_bwd = new Float32Array(this.DISPLACEMENT_BWD_SIZE);

    this.model = new PoseNet(this.tfmodel, backend, Number(this._version), 
                             Number(this._outputStride), this.scaleInputSize, this._type);   
    result = await this.model.createCompiledModel();
    console.log('compilation result: ${result}');
    this.initialized = true;
  }

  async predict(imgElement){
    if(!this.initialized){
      return;
    }
    let imageSize = [this.scaleWidth, this.scaleHeight, 3];
    let scaleData = await this.scaleImage();
    prepareInputTensor(this.inputTensor,this.scaleCanvas, this._outputStride, imageSize);
    let start = performance.now();
    let result = await this.model.computeMultiPose(this.inputTensor, this.heatmapTensor, 
                                                   this.offsetTensor, this.displacement_fwd, 
                                                   this.displacement_bwd);
    console.log("execution time: ", performance.now()-start);        
  }

  drawOutput(){    
    let imageSize = [this.scaleWidth, this.scaleHeight, 3];
    let multiPose = decodeMultiPose(this.heatmapTensor, this.offsetTensor, 
                                    this.displacement_fwd, this.displacement_bwd, 
                                    this._outputStride, this._maxDetection, this._minScore, 
                                    this._nmsRadius, toHeatmapsize(imageSize, this._outputStride));
    let singlePose = decodeSinglepose(this.heatmapTensor, this.offsetTensor, 
                                      toHeatmapsize(imageSize, this._outputStride), 
                                      this._outputStride);
    singlePose.forEach((pose)=>{
      scalePose(pose, inputSize[1]/this.scaleWidth);
      if(pose.score >= this._minScore){
        drawKeypoints(pose.keypoints, this._minScore, this.canvasContextSingle);
        drawSkeleton(pose.keypoints, this._minScore, this.canvasContextSingle);
      }
    });
    multiPose.forEach((pose)=>{
      scalePose(pose, inputSize[1]/this.scaleWidth);
      if(pose.score >= this._minScore){
        drawKeypoints(pose.keypoints, this._minScore, this.canvasContextMulti);
        drawSkeleton(pose.keypoints, this._minScore, this.canvasContextMulti);
      }
    });
  }

  scaleImage(){
    const scale = this.scaleWidth/inputSize[1];
    let pixel = this.canvasContextMulti.getImageData(0,0, inputSize[1], inputSize[2]);
    this.scaleCanvas.width = this.scaleWidth;
    this.scaleCanvas.height = this.scaleHeight;
    this.scaleCanvas.setAttribute("width", this.scaleWidth);
    this.scaleCanvas.setAttribute("height", this.scaleHeight);
    let destImg = this.scaleCtx.createImageData(this.scaleWidth, this.scaleHeight);
    const promise = new Promise((resolve, reject)=>{
      bilinear(pixel, destImg, scale);
      this.scaleCtx.putImageData(destImg, 0, 0);
      resolve(destImg.data);
    });
    return promise;
  }
}
