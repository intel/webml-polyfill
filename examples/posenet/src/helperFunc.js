function getURL(version){
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

async function getVariable(url, binary){
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
                }else{
                    reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
                }
            }
        };
        xhr.send();
    });
}


async function getDimensionData(layername, version, blockId){
    if(layername =="conv2d"){
        let manifest = await getVariable(getURL(version)+"manifest.json", false);
        manifest = JSON.parse(manifest);
        let layer = "MobilenetV1/Conv2d_"+String(blockId)+"/weights";
        let layer_bias = "MobilenetV1/Conv2d_"+String(blockId)+"/biases";
        let shape = manifest[layer]["shape"];
        let shape_bias = manifest[layer_bias]["shape"];
        let filename = manifest[layer]["filename"];
        let filename_bias = manifest[layer_bias]["filename"];
        let address = getURL(version)+filename;
        let address_bias = getURL(version)+filename_bias;
        let data = await getVariable(address, true);
        let bia = await getVariable(address_bias, true);
        const weights = new Float32Array(data);
        const bias = new Float32Array(bia);
        return [shape, weights, shape_bias, bias];
    }
    if(layername =="separableConv"){
        let manifest = await getVariable(getURL(version)+"manifest.json", false);
        manifest = JSON.parse(manifest);
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
        let data_1 = await getVariable(getURL(version)+filename1, true);
        let data_2 = await getVariable(getURL(version)+filename2, true);
        let data_1_bias = await getVariable(getURL(version)+filename1_bias, true);
        let data_2_bias = await getVariable(getURL(version)+filename2_bias, true);
        weights.push(new Float32Array(data_1));
        weights.push(new Float32Array(data_2));
        bias.push(new Float32Array(data_1_bias));
        bias.push(new Float32Array(data_2_bias));
        return [shape, weights, shape_bias, bias];
    }
}


async function getOutputLayer(layername, version){
    let manifest = await getVariable(getURL(version)+"manifest.json", false);
    manifest = JSON.parse(manifest);
    let shape;
    let shape_bias;
    let weights, bias;
    shape = manifest["MobilenetV1/"+layername+"_2/weights"]["shape"];
    shape_bias = manifest["MobilenetV1/"+layername+"_2/biases"]["shape"];
    let data = await getVariable(getURL(version)+manifest["MobilenetV1/"+layername+"_2/weights"]["filename"], true);
    weights = new Float32Array(data);
    let data_bias = await getVariable(getURL(version)+manifest["MobilenetV1/"+layername+"_2/biases"]["filename"], true);
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
    let new_weights = new Float32Array(product);
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
    if((width-1) % outputStride != 0){
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
        if(array[i] > array[max]){
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
    let y = (Math.floor(index/height))%width;
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
    let index = Number(z)+Number(x*channel)+Number(y*width*channel);
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
    if(dimension.length == 3){
        heatmapSize = [(dimension[0]-1)/outputStride+1, (dimension[1]-1)/outputStride+1, 17];
    }
    if(dimension.length == 4){
        heatmapSize = [(dimension[1]-1)/outputStride+1, (dimension[2]-1)/outputStride+1, 17]
    }
    return heatmapSize;
}

function product(array){
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

function decodeMultiPose(heatmap, offsets, displacement_fwd, displacement_bwd, outputStride, 
                         maxPoseDectection, scoreThreshold, nmsRadius, dimension){
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

function convert4D(n, h, w, c, dimension){
    let index = Number(c)+Number(w*dimension[3])+Number(h*dimension[3]*dimension[2])+
                Number(n*dimension[1]*dimension[2]*dimension[3]);
    return index;
}


function dilationWeights(weights, dimension, rate){
    let dilation_w = dimension[2]*rate-rate+1;
    let dilation_h = dimension[1]*rate-rate+1;
    let dilationweights = new Float32Array(dimension[0]*dilation_w*dilation_h*dimension[3]);
    dilationweights.fill(0);
    let dimension_dilation = [dimension[0], dilation_h, dilation_w, dimension[3]];
    for(let h = 0; h<dilation_h; h+=rate){
        for(let w = 0; w<dilation_w; w+=rate){
            for(let c = 0; c<dimension[3]; c++){
                let index_dilation = convert4D(0, h, w, c, dimension_dilation);
                let index_origin = convert4D(0, h/rate, w/rate, c, dimension);
                dilationweights[index_dilation] = weights[index_origin];
            }
        }
    }
    return [dimension_dilation, dilationweights];
}


function getValidResolution(imageScaleFactor, inputDimension, outputStride){
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
          
            // I let the r, g, b, a on purpose for debugging
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

function scalePose(pose, scale){
    for(let i in pose.keypoints){
        pose.keypoints[i].position.x = pose.keypoints[i].position.x * scale;
        pose.keypoints[i].position.y = pose.keypoints[i].position.y * scale;
    }
}