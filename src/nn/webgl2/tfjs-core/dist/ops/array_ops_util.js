"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getReshaped(inputShape, blockShape, prod, batchToSpace) {
    if (batchToSpace === void 0) { batchToSpace = true; }
    var reshaped = [];
    if (batchToSpace) {
        reshaped = reshaped.concat(blockShape.slice(0));
        reshaped.push(inputShape[0] / prod);
        reshaped = reshaped.concat(inputShape.slice(1));
    }
    else {
        reshaped = reshaped.concat(inputShape[0]);
        var spatialLength = blockShape.length;
        for (var i = 0; i < spatialLength; ++i) {
            reshaped =
                reshaped.concat([inputShape[i + 1] / blockShape[i], blockShape[i]]);
        }
        reshaped = reshaped.concat(inputShape.slice(spatialLength + 1));
    }
    return reshaped;
}
exports.getReshaped = getReshaped;
function getPermuted(reshapedRank, blockShapeRank, batchToSpace) {
    if (batchToSpace === void 0) { batchToSpace = true; }
    var permuted = [];
    if (batchToSpace) {
        permuted.push(blockShapeRank);
        for (var i = blockShapeRank + 1; i < reshapedRank; ++i) {
            if (i <= 2 * blockShapeRank) {
                permuted.push(i);
                permuted.push(i - (blockShapeRank + 1));
            }
            else {
                permuted.push(i);
            }
        }
    }
    else {
        var permutedBeforeBatch = [];
        var permutedAfterBatch = [];
        for (var i = 1; i < reshapedRank; ++i) {
            if (i >= blockShapeRank * 2 + 1 || i % 2 === 1) {
                permutedAfterBatch.push(i);
            }
            else {
                permutedBeforeBatch.push(i);
            }
        }
        permuted.push.apply(permuted, permutedBeforeBatch);
        permuted.push(0);
        permuted.push.apply(permuted, permutedAfterBatch);
    }
    return permuted;
}
exports.getPermuted = getPermuted;
function getReshapedPermuted(inputShape, blockShape, prod, batchToSpace) {
    if (batchToSpace === void 0) { batchToSpace = true; }
    var reshapedPermuted = [];
    if (batchToSpace) {
        reshapedPermuted.push(inputShape[0] / prod);
    }
    else {
        reshapedPermuted.push(inputShape[0] * prod);
    }
    for (var i = 1; i < inputShape.length; ++i) {
        if (i <= blockShape.length) {
            if (batchToSpace) {
                reshapedPermuted.push(blockShape[i - 1] * inputShape[i]);
            }
            else {
                reshapedPermuted.push(inputShape[i] / blockShape[i - 1]);
            }
        }
        else {
            reshapedPermuted.push(inputShape[i]);
        }
    }
    return reshapedPermuted;
}
exports.getReshapedPermuted = getReshapedPermuted;
function getSliceBeginCoords(crops, blockShape) {
    var sliceBeginCoords = [0];
    for (var i = 0; i < blockShape; ++i) {
        sliceBeginCoords.push(crops[i][0]);
    }
    return sliceBeginCoords;
}
exports.getSliceBeginCoords = getSliceBeginCoords;
function getSliceSize(uncroppedShape, crops, blockShape) {
    var sliceSize = uncroppedShape.slice(0, 1);
    for (var i = 0; i < blockShape; ++i) {
        sliceSize.push(uncroppedShape[i + 1] - crops[i][0] - crops[i][1]);
    }
    return sliceSize;
}
exports.getSliceSize = getSliceSize;
//# sourceMappingURL=array_ops_util.js.map