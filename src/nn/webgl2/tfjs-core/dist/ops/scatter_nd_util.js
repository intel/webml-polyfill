"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util_1 = require("../util");
function validateUpdateShape(shape, indices, updates) {
    var sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
    var batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;
    var shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
        ("shape[sliceDim:], got updates.shape: " + updates.shape) +
        (", indices.shape: " + indices.shape + ", shape: " + shape) +
        (", sliceDim: " + sliceDim + ", and batchDim: " + batchDim + ".");
    if (updates.rank < batchDim) {
        throw new Error(shapeError + (" update.rank < " + batchDim + ". "));
    }
    if (shape.length < sliceDim + (updates.rank - batchDim)) {
        throw new Error(shapeError +
            (" Output shape length < " + (sliceDim + (updates.rank - batchDim))));
    }
    if (updates.rank !== batchDim + shape.length - sliceDim) {
        throw new Error(shapeError + (" update.rank != " + (batchDim + shape.length - sliceDim)));
    }
    for (var d = 0; d < batchDim; ++d) {
        if (updates.shape[d] !== indices.shape[d]) {
            throw new Error(shapeError +
                (" updates.shape[" + d + "] (" + updates.shape[d] + ") != indices.shape[" + d + "] (" + indices.shape[d] + ")."));
        }
    }
    for (var d = 0; d < updates.rank - batchDim; ++d) {
        if (updates.shape[d + batchDim] !== shape[d + sliceDim]) {
            throw new Error(shapeError +
                (" updates.shape[" + (d + batchDim) + "] (" + updates.shape[d + batchDim] + ") != shape[" + (d + batchDim) + "] (" + shape[d + batchDim] + ")"));
        }
    }
}
exports.validateUpdateShape = validateUpdateShape;
function validateInput(updates, indices, shape) {
    if (indices.rank < 1) {
        throw new Error('tf.scatterND() expects the indices to be rank 1 or higher,' +
            (" but the rank was " + indices.rank + "."));
    }
    if (updates.rank < 1) {
        throw new Error('tf.scatterND() expects the updates to be rank 1 or higher,' +
            (" but the rank was " + updates.rank + "."));
    }
    if (indices.dtype !== 'int32') {
        throw new Error("The dtype of 'indices' should be int32, but got dtype: " + indices.dtype);
    }
    if (shape.length < 1) {
        throw new Error("Output rank must be greater or equal to 1, but got shape: " + shape);
    }
    if (shape.length === 0) {
        if (indices.size === 0) {
            throw new Error("Indices specified for empty output. indices shape: " + indices.shape);
        }
        if (updates.size === 0) {
            throw new Error("Updates specified for empty output. updates shape: " + updates.shape);
        }
    }
    validateUpdateShape(shape, indices, updates);
}
exports.validateInput = validateInput;
function calculateShapes(updates, indices, shape) {
    var sliceRank = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
    var totalNd = shape.length;
    var sliceSize = 1;
    for (var i = sliceRank; i < totalNd; ++i) {
        sliceSize *= shape[i];
    }
    var safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
    var numUpdates = indices.size / safeSliceDim;
    var outputStrides = util_1.computeStrides(shape).concat([1]);
    var strides = outputStrides.slice(outputStrides.length - sliceRank, outputStrides.length);
    var outputSize = util_1.sizeFromShape(shape);
    return { sliceRank: sliceRank, numUpdates: numUpdates, sliceSize: sliceSize, strides: strides, outputSize: outputSize };
}
exports.calculateShapes = calculateShapes;
//# sourceMappingURL=scatter_nd_util.js.map