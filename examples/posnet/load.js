const version = 1.01|1.0|0.75|0.5;

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
const HEATMAP_TENSOR_SIZE = 33*33*17;
const OFFSET_TENSOR_SIZE = 33*33*34;
const DISPLACEMENT_FWD_SIZE = 33*33*32;
const DISPLACEMENT_BWD_SIZE = 33*33*32;
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
        this._outputStride;

        this.inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
        this.heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
        this.offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
        this.displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
        this.displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
        //single input
        this._version = document.getElementById('modelversion').value;
        this._outputStride= document.getElementById('outputStride').value;
        this._minScore = document.getElementById('minpartConfidenceScore').value;
        //multiple input
        this._nmsRadius = document.getElementById('nmsRadius').value;
        this._maxDetection = document.getElementById('maxDetection').value;

        this.canvasElement_single = document.getElementById('canvas');
        //this.canvasContext = this.canvasElement.getContext('2d');
        this.canvasElement_multi = document.getElementById('canvas_2');
        this._type = "Multiperson";
        this.initialized = false;
    }

    async init(backend){
        this.initialized = false;
        let result;
        let variable;

        if(!this.tfmodel){
            const ModelArch = new Map([
                [0.5, mobileNet50Architecture],
                [0.75, mobileNet75Architecture],
                [1.0, mobileNet100Architecture],
                [1.01, mobileNet100Architecture],
            ]);
            this.tfmodel = ModelArch.get(this._version);
        }
        this.model = new PoseNet(this.tfmodel, backend, this._version, this._outputStride, input_size, this._type);
        result = await this.model.createCompiledModel();
        console.log('compilation result: ${result}');
        this.initialized = true;
    }

    async predict(imageSource){
        if(!this.initialized){
            return;
        }
        imageSource.drawTo(this.canvasElement_single);
        imageSource.drawTo(this.canvasElement_multi);


    }

    async loadmanifest(url){
        var address = url+"manifest.json";
        return fetch(address)
        .then(function(response) {
            return response.json();
        })
        .then(function(myJson) {
            var data = {};
            for(var i in myJson){
                data[i] = myJson[i];
            } 
            return data;
        });
    }

    async getvariable(url, binary){
        return new Promise(function(resolve, reject){
            var xhr = new XMLHttpRequest();
            xhr.open("GET", url, true);
            if(binary){
                xhr.responseType = 'arraybuffer';
            }
            xhr.onload = function(ev){
                if(xhr.readyState == 4){
                    if(xhr.status == 200){
                        resolve(xhr.response);
                    }
                    else{
                        reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
                    }
                }
            };
            xhr.send();
        });
    }


    getURL(version){
        let address;
        switch(version){
            case 1.01:
                address = 'https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101/';
                break;
            case 1.0:
                address = 'https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_100/';
                break;
            case 0.75:
                address = 'https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_075/';
                break;
            case 0.5:
                address = 'https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_050/';
                break;
            default:
                console.log("It must be 1.01, 1.0, 0.75 or 0.5");
        }
        return address;
    }

}

    





// function Layer(blockId, stride, outputStride, convType, rate){
//     this.blockId = blockId;
//     this.stride = stride;
//     this.outputStride = outputStride;
//     this.convType = convType;
//     this.rate = rate;
// }

// for(var j in mobileNet100Architecture){
//     var layer = new Layer(j, mobileNet100Architecture[j][1], 16, mobileNet100Architecture[j][0], 1);
//     if(layer.convType == 'conv2d'){
//         manifestload(model(1.0)).then(function(mobilenet){
            
//         })
//     }
// }


// var temp = manifestload(model(1.01));
// temp.then(function(mobilenet){
//     var blockid = 10;
//     var name = "MobilenetV1/Conv2d_"+String(blockid)+"_depthwise/depthwise_weights";
//     //console.log(mobilenet[name]["filename"]);
//     var shape = mobilenet[name]["shape"];
//     console.log(shape);
//     var fileaddress = model(1.01)+mobilenet[name]["filename"];
//     getvariable(fileaddress).then(function(data){
//         const values = new Int32Array(data);
//         console.log(values);
//     })
// });



