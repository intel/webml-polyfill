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

//Architecture = [layer name, stride]
//conv2d: convolution layer
//separableConv: depthwise convolution layer + pointwise convolution layer
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

const INPUT_TENSOR_SIZE = 513*513*3;
const input_size = [1, 513, 513, 3];


class Utils{
    constructor(){
        this.tfmodel;
        this.model;
        this.inputTensor;
        this.heatmapTensor;
        this.offsetTensor;
        this.displacement_fwd;
        this.displacement_bwd;
        this._version = 1.01;

        //single input
        this._version = document.getElementById('modelversion').value;
        this._outputStride= document.getElementById('outputStride').value;
        this._minScore = document.getElementById('minpartConfidenceScore').value;
        //multiple input
        this._nmsRadius = document.getElementById('nmsRadius').value;
        this._maxDetection = document.getElementById('maxDetection').value;
        //image Source
        this._inputElement = document.getElementById('image').files[0];
        
        this.canvasElement_single = document.getElementById('canvas');
        this.canvasContext_single = this.canvasElement_single.getContext('2d');
        this.canvasElement_multi = document.getElementById('canvas_2');
        this.canvasContext_multi = this.canvasElement_multi.getContext('2d');
        this._type = "Multiperson";
        this.initialized = false;

        if(this._outputStride.length == 0){
            this._outputStride = 16;
        }

        this.HEATMAP_TENSOR_SIZE = product(toHeatmapsize(input_size, this._outputStride));
        this.OFFSET_TENSOR_SIZE = this.HEATMAP_TENSOR_SIZE*2;
        this.DISPLACEMENT_FWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;
        this.DISPLACEMENT_BWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;

        this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
        this.heatmapTensor = new Float32Array(this.HEATMAP_TENSOR_SIZE);
        this.offsetTensor = new Float32Array(this.OFFSET_TENSOR_SIZE);
        this.displacement_fwd = new Float32Array(this.DISPLACEMENT_FWD_SIZE);
        this.displacement_bwd = new Float32Array(this.DISPLACEMENT_BWD_SIZE);
    }

    async init(backend){
        this.initialized = false;
        let result;
        let variable;
        if(this._minScore<0 | this._minScore>1){
            alert("Minimal Part Confidence Score must be in range (0,1).");
            return;
        }
        if(this._outputStride!=8 & this._outputStride!=16 & this._outputStride!=32){
            alert("OutputSride must be 8, 16 or 32");
            return;
        }
        if(!this.tfmodel){
            var ModelArch = new Map([
                [0.5, mobileNet50Architecture],
                [0.75, mobileNet75Architecture],
                [1.0, mobileNet100Architecture],
                [1.01, mobileNet100Architecture],
            ]);
            this.tfmodel = ModelArch.get(Number(this._version));
        }
        this.model = new PoseNet(this.tfmodel, backend, Number(this._version), 
                        Number(this._outputStride), input_size, this._type);
        result = await this.model.createCompiledModel();
        console.log('compilation result: ${result}');
        this.initialized = true;
    }

    async predict(){
        if(!this.initialized){
            return;
        }

        this.canvasContext_single.clearRect(0, 0, this.canvasElement_single.width, this.canvasElement_single.height);
        this.canvasContext_multi.clearRect(0, 0, this.canvasElement_multi.width, this.canvasElement_multi.height);
        if(this._inputElement!=undefined){
            let x = await getInput(this._inputElement);
            await loadImage(x, this.canvasContext_single);
            await loadImage(x, this.canvasContext_multi);
        }else{
            await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", this.canvasContext_multi);
            await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", this.canvasContext_single);
        }
        let imageSize = [input_size[1], input_size[2], input_size[3]];
        prepareInputTensor(this.inputTensor,this.canvasElement_multi, this._outputStride, imageSize);
        let start = performance.now();
        let result = await this.model.compute_multi(this.inputTensor, this.heatmapTensor, 
                this.offsetTensor, this.displacement_fwd, this.displacement_bwd);
        console.log("execution time: ", performance.now()-start);
        let poses_multi = decodeMultiPose(this.heatmapTensor, this.offsetTensor, 
                        this.displacement_fwd, this.displacement_bwd, 
                        this._outputStride, this._maxDetection, this._minScore, 
                        this._nmsRadius, toHeatmapsize(imageSize, this._outputStride));
        let poses_single = decodeSinglepose(this.heatmapTensor, this.offsetTensor, 
                            toHeatmapsize(imageSize, this._outputStride), this._outputStride);
        poses_single.forEach((pose)=>{
            if(pose.score>= this._minScore, this.canvasContext_single){
                drawKeypoints(pose.keypoints, this._minScore, this.canvasContext_single);
                drawSkeleton(pose.keypoints, this._minScore, this.canvasContext_single);
            }
        });

        poses_multi.forEach((pose)=>{
            if(pose.score>= this._minScore){
                drawKeypoints(pose.keypoints, this._minScore, this.canvasContext_multi);
                drawSkeleton(pose.keypoints, this._minScore, this.canvasContext_multi);
            }
        });
    }
}