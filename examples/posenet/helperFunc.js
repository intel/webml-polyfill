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
        let manifest = await util.loadmanifest(util.getURL(version));
        let layer = "MobilenetV1/Conv2d_"+String(blockId)+"/weights";
        let layer_bias = "MobilenetV1/Conv2d_"+String(blockId)+"/biases";
        let shape = manifest[layer]["shape"];
        let shape_bias = manifest[layer_bias]["shape"];
        let filename = manifest[layer]["filename"];
        let filename_bias = manifest[layer_bias]["filename"];
        let address = util.getURL(version)+filename;
        let address_bias = util.getURL(version)+filename_bias;
        let data = await util.getvariable(address, true);
        let bia = await util.getvariable(address_bias, true);
        const weights = new Float32Array(data);
        const bias = new Float32Array(bia);
        return [shape, weights, shape_bias, bias];
    }
    if(layername=="separableConv"){
        let manifest = await util.loadmanifest(util.getURL(version));
        let layer_1 = "MobilenetV1/Conv2d_"+String(blockId)+"_depthwise/depthwise_weights";
        let layer_2 = "MobilenetV1/Conv2d_"+String(blockId)+"_pointwise/weights";
        let layer_1_bias = "MobilenetV1/Conv2d_"+String(blockId)+"_depthwise/biases";
        let layer_2_bias = "MobilenetV1/Conv2d_"+String(blockId)+"_pointwise/biases";
        let shape = [];
        let shape_bias = [];
        let weights = [];
        let bias = [];
        shape.push(manifest[layer_1]["shape"]);
        shape.push(manifest[layer_2]["shape"]);
        shape_bias.push(manifest[layer_1_bias]["shape"]);
        shape_bias.push(manifest[layer_2_bias]["shape"]);
        let filename1 = manifest[layer_1]["filename"];
        let filename2 = manifest[layer_2]["filename"];
        let filename1_bias = manifest[layer_1_bias]["filename"];
        let filename2_bias = manifest[layer_2_bias]["filename"];
        let data_1 = await util.getvariable(util.getURL(version)+filename1, true);
        let data_2 = await util.getvariable(util.getURL(version)+filename2, true);
        let data_1_bias = await util.getvariable(util.getURL(version)+filename1_bias, true);
        let data_2_bias = await util.getvariable(util.getURL(version)+filename2_bias, true);
        weights.push(new Float32Array(data_1));
        weights.push(new Float32Array(data_2));
        bias.push(new Float32Array(data_1_bias));
        bias.push(new Float32Array(data_2_bias));
        return [shape, weights, shape_bias, bias];
    }
}

async function getOutputLayer(layername, version){
    let util = new Utils();
    let manifest = await util.loadmanifest(util.getURL(version));
    let shape;
    let shape_bias;
    let weights, bias;
    shape = manifest["MobilenetV1/"+layername+"_2/weights"]["shape"];
    shape_bias = manifest["MobilenetV1/"+layername+"_2/biases"]["shape"];
    let data = await util.getvariable(util.getURL(version)+manifest["MobilenetV1/"+layername+"_2/weights"]["filename"], true);
    weights = new Float32Array(data);
    let data_bias = await util.getvariable(util.getURL(version)+manifest["MobilenetV1/"+layername+"_2/biases"]["filename"], true);
    bias = new Float32Array(data_bias);
    return [shape, weights, shape_bias, bias];
}

//obtain desired size output
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
    let new_dimension = [];
    new_dimension.push(dimension[dimension.length-1])
    for(let i =0; i<dimension.length-1; i++){
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

function prepareInputTensor(tensor, canvas, outputStride, img_dimension){
    const width = img_dimension[0];
    const height = img_dimension[1]; 
    const channels = img_dimension[2];
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
    for(let i in heatmap){
        heatmap[i] = 1/(1+Math.pow(Math.E, -heatmap[i]));
    }
    return heatmap;
}

function argMax(array){
    let max = 0;
    for(let i in array){
        if(array[i]>array[max]){
            max = i;
        }
    }
    return max;
}


function getKeypointIndex(array, dimension){
    let new_array = [];
    let index = [];
    let confidenceScore = [];
    for(let i = 0; i< 17; i++){
        new_array.push(new Array(dimension[0]*dimension[1]));
    }
    for(let i in array){
        new_array[i%17][Math.floor(i/17)] = array[i];
    }
    for(let j in new_array){
        index.push(argMax(new_array[j]));
        confidenceScore.push(new_array[j][argMax(new_array[j])]);
    }
    return [index, confidenceScore];
}


function convertPosition(index, dimension){
    let height = dimension[0];
    let width = dimension[1];
    let x = index%height;
    let y = (Math.floor(index/height)) % width;
    return [y, x];
}

function convertIndextoCoor(index, dimension){
    [height, width, channel] = dimension;
    let z = index%channel;
    let x = Math.floor(index/channel)%width;
    let y = Math.floor(index/(width*channel));
    return [y, x, z];
}

function convertCoortoIndex(x, y, z, dimension){
    let [height, width, channel] = dimension;
    let index = Number(z) + Number(x*channel) + Number(y*width*channel);
    return index;
}

function decodeSinglepose(heatmap, offset, dimension, outputStride){
    let [index, confidenceScore] = getKeypointIndex(sigmoid(heatmap),dimension);
    let final_res = [];
    let total_score = 0; 
    let poses = [];
    for(let i in index){
        let heatmap_y = convertPosition(Number(index[i]), dimension)[0];
        let heatmap_x = convertPosition(Number(index[i]), dimension)[1];
        let offset_y = offset[Number(i)+Number(heatmap_x)*34+Number(heatmap_y*34*dimension[1])];
        let offset_x = offset[Number(i)+17+Number(heatmap_x)*34+Number(heatmap_y*34*dimension[1])];
        let final_pos = [heatmap_y*outputStride+offset_y, heatmap_x*outputStride+offset_x];
        total_score += confidenceScore[i];
        final_res.push({position: {y: final_pos[0], x: final_pos[1]}, 
                    part: partNames[i], score: confidenceScore[i]});
    }
    poses.push({keypoints: final_res, score: total_score/17});
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
        return squaredDistance(y, x, correspondingKeypoint.y, correspondingKeypoint.x) <=
            squaredNmsRadius;
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

function toHeatmapsize(dimension, outputStride){
    let heatmapSize;
    if(dimension.length==3){
        heatmapSize = [(dimension[0]-1)/outputStride+1, (dimension[1]-1)/outputStride+1, 17];
    }
    if(dimension.length==4){
        heatmapSize = [(dimension[1]-1)/outputStride+1, (dimension[2]-1)/outputStride+1, 17]
    }
    return heatmapSize;
}

function Product(array){
    return array.reduce(function(a,b){return a*b;});
}


function buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores, dimension){
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
                // if (score < scoreThreshold) {
                //     continue;
                // }
                if (scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores, dimension)) {
                    queue.enqueue({ score: score, part: { heatmapY: heatmapY, heatmapX: heatmapX, id: keypointId } });
                }
            }
        }
    }
    return queue;
}

function getImageCoords(part, outputStride, offsets, dimension){
    dimension_offset = [];
    dimension_offset.push(dimension[0]);
    dimension_offset.push(dimension[1]);
    dimension_offset.push(34);
    let heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
    let index_y = convertCoortoIndex(heatmapX, heatmapY, keypoint, dimension_offset);
    let index_x = index_y+17;
    return {
        x: part.heatmapX * outputStride + offsets[index_x],
        y: part.heatmapY * outputStride + offsets[index_y]
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

function decodeMultiPose(heatmap, offsets, displacement_fwd, displacement_bwd, 
                        outputStride, maxPoseDectection, scoreThreshold, nmsRadius, dimension){
    let poses = [];
    let queue = buildPartWithScoreQueue(scoreThreshold, 1, sigmoid(heatmap), dimension);
    let _root, keypoints, score;
    const squaredNmsRadius = nmsRadius * nmsRadius;
    let index = 0;
    while(poses.length < maxPoseDectection && !queue.empty()){
        _root = queue.dequeue();
        rootImgCoord = getImageCoords(_root.part, outputStride, offsets, dimension);
        if(withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImgCoord, _root.part.id)){
            continue;
        }
        keypoints = decodePose(_root, heatmap, offsets, outputStride, displacement_fwd, displacement_bwd, dimension);
        score = getInstanceScore(poses, squaredNmsRadius, keypoints);
        poses.push({keypoints: keypoints, score: score});
    }
    return poses;
}   

