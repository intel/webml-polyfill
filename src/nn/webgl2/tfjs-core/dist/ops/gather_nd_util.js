"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util_1 = require("../util");
function prepareAndValidate(tensor, indices) {
    if (tensor.rank < 1) {
        throw new Error('tf.gatherND() expects the input to be rank 1 or higher,' +
            (" but the rank was " + tensor.rank + "."));
    }
    if (indices.rank < 1) {
        throw new Error('tf.gatherND() expects the indices to be rank 1 or higher,' +
            (" but the rank was " + indices.rank + "."));
    }
    if (indices.dtype !== 'int32') {
        throw new Error('tf.gatherND() expects the indices to be int32 type,' +
            (" but the dtype was " + indices.dtype + "."));
    }
    if (indices.shape[indices.rank - 1] > tensor.rank) {
        throw new Error('index innermost dimension length must be <= tensor rank; saw: ' +
            (indices.shape[indices.rank - 1] + " vs. " + tensor.rank));
    }
    if (tensor.size === 0) {
        throw new Error('Requested more than 0 entries, but input is empty.' +
            (" Input shape: " + tensor.shape + "."));
    }
    var indicesShape = indices.shape;
    var sliceRank = indicesShape[indicesShape.length - 1];
    var nResult = 1;
    for (var i = 0; i < indicesShape.length - 1; ++i) {
        nResult *= indicesShape[i];
    }
    var inputShape = tensor.shape;
    var resultShape = indicesShape.slice();
    resultShape.pop();
    var sliceSize = 1;
    for (var i = sliceRank; i < tensor.rank; ++i) {
        sliceSize *= inputShape[i];
        resultShape.push(inputShape[i]);
    }
    var strides = util_1.computeStrides(tensor.shape).map(function (stride) { return stride / sliceSize; }).concat([1]).slice(0, sliceRank);
    return [resultShape, nResult, sliceSize, strides];
}
exports.prepareAndValidate = prepareAndValidate;
//# sourceMappingURL=gather_nd_util.js.map