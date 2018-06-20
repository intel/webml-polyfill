let Architecture = [
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
];
  

async function getDimension_data(layername, version, blockId){
    let util = new Utils();
    if(layername=="conv2d"){
        var manifest = await util.loadmanifest(util.getURL(version));
        var layer = "MobilenetV1/Conv2d_"+String(blockId)+"/weights";
        var layer_bias = "MobilenetV1/Conv2d_"+String(blockId)+"/biases";
        var shape = manifest[layer]["shape"];
        var shape_bias = manifest[layer_bias]["shape"];
        var filename = manifest[layer]["filename"];
        var filename_bias = manifest[layer_bias]["filename"];
        var address = util.getURL(version)+filename;
        var address_bias = util.getURL(version)+filename_bias;
        var data = await util.getvariable(address, true);
        var bia = await util.getvariable(address_bias, true);
        const weights = new Float32Array(data);
        const bias = new Float32Array(bia);
        //console.log(weights);
        return [shape, weights, shape_bias, bias];
    }
    if(layername=="separableConv"){
        var manifest = await util.loadmanifest(util.getURL(version));
        var layer_1 = "MobilenetV1/Conv2d_"+String(blockId)+"_depthwise/depthwise_weights";
        var layer_2 = "MobilenetV1/Conv2d_"+String(blockId)+"_pointwise/weights";
        var layer_1_bias = "MobilenetV1/Conv2d_"+String(blockId)+"_depthwise/biases";
        var layer_2_bias = "MobilenetV1/Conv2d_"+String(blockId)+"_pointwise/biases";
        var shape = [];
        var shape_bias = [];
        var weights = [];
        var bias = [];
        shape.push(manifest[layer_1]["shape"]);
        shape.push(manifest[layer_2]["shape"]);
        shape_bias.push(manifest[layer_1_bias]["shape"]);
        shape_bias.push(manifest[layer_2_bias]["shape"]);
        var filename1 = manifest[layer_1]["filename"];
        var filename2 = manifest[layer_2]["filename"];
        var filename1_bias = manifest[layer_1_bias]["filename"];
        var filename2_bias = manifest[layer_2_bias]["filename"];
        var data_1 = await util.getvariable(util.getURL(version)+filename1, true);
        var data_2 = await util.getvariable(util.getURL(version)+filename2, true);
        var data_1_bias = await util.getvariable(util.getURL(version)+filename1_bias, true);
        var data_2_bias = await util.getvariable(util.getURL(version)+filename2_bias, true);
        weights.push(new Float32Array(data_1));
        weights.push(new Float32Array(data_2));
        bias.push(new Float32Array(data_1_bias));
        bias.push(new Float32Array(data_2_bias));
        return [shape, weights, shape_bias, bias];
    }
}

async function getOutputLayer(layername, version){
    let util = new Utils();
    var manifest = await util.loadmanifest(util.getURL(version));
    var shape;
    var shape_bias;
    var weights, bias;
    shape = manifest["MobilenetV1/"+layername+"_2/weights"]["shape"];
    shape_bias = manifest["MobilenetV1/"+layername+"_2/biases"]["shape"];
    var data = await util.getvariable(util.getURL(version)+manifest["MobilenetV1/"+layername+"_2/weights"]["filename"], true);
    weights = new Float32Array(data);
    var data_bias = await util.getvariable(util.getURL(version)+manifest["MobilenetV1/"+layername+"_2/biases"]["filename"], true);
    bias = new Float32Array(data_bias);
    return [shape, weights, shape_bias, bias];
}

//obtain desired size output
function toOutputStridedLayers(convolutionDefinition, outputStride) {
    var currentStride = 1;
    var rate = 1;
    return convolutionDefinition.map(function (_a, blockId) {
        var convType = _a[0], stride = _a[1];
        var layerStride, layerRate;
        if (currentStride === outputStride) {
            layerStride = 1;
            layerRate = rate;
            rate *= stride;
        }
        else {
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

function resize(dimension){
    var new_dimension = [];
    new_dimension.push(dimension[dimension.length-1])
    for(var i =0; i<dimension.length-1; i++){
        new_dimension.push(dimension[i]);
    }
    return new_dimension;
}

function transpose_weights(weights, dimension){
    let product = dimension.reduce(function(a,b){return a*b});
    let new_weights = new Array(product);
    let [H, W, C, N] = dimension;
    for(let h =0; h<H; h++){
        for(let w =0; w<W; w++){
            for(let c = 0; c<C; c++){
                for(let n =0; n<N; n++){
                    new_weights[c + w*C + h*W*C + n*H*W*C] = 
                        weights[n + c*N + w*C*N + h*W*C*N];
                }
            }
        }
    }
    return new_weights;
}

function valideResolution(input_dimension, outputStride){
    let width = input_dimension[1];
    let height = input_dimension[2];
    if((width-1)%outputStride!=0){
        throw new Error("invalid resolution");
    }
}

function prepareInputTensor(tensor, canvas, outputStride){
    const width = 513;
    const height = 513; 
    const channels = 3;
    const imageChannels = 4;
    const mean = 127.5;
    const std = 127.5;
    let dimension = [1, canvas.width, canvas.height, 3];
    valideResolution(dimension, outputStride);
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

function sigmoid(heatmap){
    for(var i in heatmap){
        heatmap[i] = 1/(1+Math.pow(Math.E, -heatmap[i]));
    }
    return heatmap;
}

function argMax(array){
    let max = 0;
    for(var i in array){
        if(array[i]>array[max]){
            max = i;
        }
    }
    return max;
}

function getKeypointIndex(array, dimension){
    var size = dimension[0]*dimension[1];
    var index = [];
    for(var i = 0; i<17; i++){
        var temp = array.subarray(i*size, (i+1)*size);
        index.push(argMax(temp));
        console.log(array[argMax(temp)]);
    }
    return index;
}

function convertPosition(index, dimension){
    let width = dimension[0];
    let height = dimension[1];
    let x = index%width;
    let y = (index/width) % height;
    return [x, y];
}

function decodeSinglepose(heatmap, offset, dimension, outputStride){
    var index = getKeypointIndex(sigmoid(heatmap),dimension);
    console.log(index);
    //console.log(offset);
    for(var i in index){
        var heatmap_x = convertPosition(Number(index[i]), dimension)[0];
        var heatmap_y = convertPosition(Number(index[i]), dimension)[1];
        var offset_y = offset[Number(index[i])+225*i];
        var offset_x = offset[Number(index[i])+225*i+3825];
        //console.log(offset_x);
        var final_pos = [heatmap_y*outputStride+offset_y, heatmap_x*outputStride+offset_x];
        console.log(final_pos);
    }
}

// let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
// var canvas = document.getElementById("canvas");
// var ctx = canvas.getContext("2d");
// var img = new Image();
// img.onload = function(){
//     ctx.drawImage(img, 0, 0);
//     let pixels = ctx.getImageData(0, 0, 225, 225);
//     console.log(pixels);
//     prepareInputTensor(inputTensor, canvas, 16);
// };
// img.src = 'people-img3.jpg';

//prepareInputTensor(inputTensor, canvas, 16);


// var x = resize([1, 1, 1024, 17]);
// console.log(x);

// getOutputLayer("offset", 1.01).then(function(data){
//     console.log(data);
//     var value = transpose_weights(data[1], data[0]);
//     console.log(value);
// });
// async function test(){
//     let util = new Utils();
//     await getDimension_data("conv2d", 1.01, 0).then(function(data){
//         console.log(data[1]);
//         var new_data = transpose_weights(data[1], [3, 3, 3, 32]);
//         console.log(new_data);
//     });
// }

// transpose([1,1,1,1,1,1,1,1,1,1], [3, 3, 3, 32]);
//console.log(resize([3, 3, 3, 32]));
// var res = toOutputStridedLayers(Architecture, 16);
// console.log(res);
// getDimension_data("separableConv", 1.01, 2).then(function(data){
//     console.log(data[2]);
//     console.log(data[3]);
// });
