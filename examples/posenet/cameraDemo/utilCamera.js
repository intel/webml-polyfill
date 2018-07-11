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
const videoWidth = 500;
const videoHeight = 500;

class Utils{
    constructor(){
        this.tfmodel;
        this.model;
        this.inputTensor;
        this.heatmapTensor;
        this.offsetTensor;
        this.displacement_fwd;
        this.displacement_bwd;
        this._version;

        //single input
        this._version = document.getElementById('modelversion').value;
        this._outputStride= document.getElementById('outputStride').value;
        this._minScore = document.getElementById('minpartConfidenceScore').value;
        this._scaleFactor = document.getElementById('scaleFactor').value;
        //multiple input
        this._nmsRadius = document.getElementById('nmsRadius').value;
        this._maxDetection = document.getElementById('maxDetection').value;
        //image Source
        
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.scaleCanvas = document.getElementById('canvas_2');
        this.scaleCtx = this.scaleCanvas.getContext('2d');
        this._type = "Multiperson";
        this.initialized = false;

        this.scaleWidth = getValidResolution(this._scaleFactor, videoWidth, this._outputStride);
        this.scaleHeight = getValidResolution(this._scaleFactor, videoHeight, this._outputStride);

        this.inputSize = [1, this.scaleWidth, this.scaleHeight, 3];
        this.INPUT_TENSOR_SIZE = this.scaleWidth*this.scaleHeight*3;
        this.HEATMAP_TENSOR_SIZE = product(toHeatmapsize(this.inputSize, this._outputStride));
        this.OFFSET_TENSOR_SIZE = this.HEATMAP_TENSOR_SIZE*2;
        this.DISPLACEMENT_FWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;
        this.DISPLACEMENT_BWD_SIZE = this.HEATMAP_TENSOR_SIZE/17*32;

        this.inputTensor = new Float32Array(this.INPUT_TENSOR_SIZE);
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
                        Number(this._outputStride), this.inputSize, this._type);
        result = await this.model.createCompiledModel();
        console.log('compilation result: ${result}');
        this.initialized = true;
    }

    scaleImage(){
        let scale = this.scaleWidth/videoWidth;
        let pixel = this.ctx.getImageData(0, 0, videoWidth, videoHeight);
        this.scaleCanvas.width = this.scaleWidth;
        this.scaleCanvas.height = this.scaleHeight;
        this.scaleCanvas.setAttribute("width", this.scaleWidth);
        this.scaleCanvas.setAttribute("height", this.scaleHeight);
        var destImg = this.scaleCtx.createImageData(this.scaleWidth, this.scaleHeight);
        const promise = new Promise((resolve, reject)=>{
            bilinear(pixel, destImg, scale);
            this.scaleCtx.putImageData(destImg, 0, 0);
            resolve(destImg.data);
        });
        return promise;
    }

    async predict(){
        if(!this.initialized){
            return;
        }
        let imageSize = [this.scaleWidth, this.scaleHeight, 3];
        let scaleData = await this.scaleImage();
        prepareInputTensor(this.inputTensor,this.scaleCanvas, this._outputStride, imageSize);
        let result = await this.model.compute_multi(this.inputTensor, this.heatmapTensor, 
                this.offsetTensor, this.displacement_fwd, this.displacement_bwd);
        let poses_multi = decodeMultiPose(this.heatmapTensor, this.offsetTensor, 
                        this.displacement_fwd, this.displacement_bwd, 
                        this._outputStride, this._maxDetection, this._minScore, 
                        this._nmsRadius, toHeatmapsize(imageSize, this._outputStride));
        let poses_single = decodeSinglepose(this.heatmapTensor, this.offsetTensor, 
                            toHeatmapsize(imageSize, this._outputStride), this._outputStride);
        // poses_single.forEach((pose)=>{
        //     if(pose.score>= this._minScore, this.canvasContext_single){
        //         drawKeypoints(pose.keypoints, this._minScore, this.canvasContext_single);
        //         drawSkeleton(pose.keypoints, this._minScore, this.canvasContext_single);
        //     }
        // });

        poses_multi.forEach((pose)=>{
            scalePose(pose, videoWidth/this.scaleWidth);
            if(pose.score>= this._minScore){
                drawKeypoints(pose.keypoints, this._minScore, this.canvas);
                drawSkeleton(pose.keypoints, this._minScore, this.canvas);
            }
        });
    }
}