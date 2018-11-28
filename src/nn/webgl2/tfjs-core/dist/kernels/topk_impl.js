"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_ops_1 = require("../ops/tensor_ops");
var util_1 = require("../util");
function topkImpl(x, xShape, xDtype, k, sorted) {
    var lastDim = xShape[xShape.length - 1];
    var _a = [x.length / lastDim, lastDim], batch = _a[0], size = _a[1];
    var allTopKVals = util_1.getTypedArrayFromDType(xDtype, batch * k);
    var allTopKIndices = util_1.getTypedArrayFromDType('int32', batch * k);
    for (var b = 0; b < batch; b++) {
        var offset = b * size;
        var vals = x.subarray(offset, offset + size);
        var valAndInd = [];
        for (var i = 0; i < vals.length; i++) {
            valAndInd.push({ value: vals[i], index: i });
        }
        valAndInd.sort(function (a, b) { return b.value - a.value; });
        var outOffset = b * k;
        var topKVals = allTopKVals.subarray(outOffset, outOffset + k);
        var topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
        for (var i = 0; i < k; i++) {
            topKVals[i] = valAndInd[i].value;
            topKIndices[i] = valAndInd[i].index;
        }
    }
    var outputShape = xShape.slice();
    outputShape[outputShape.length - 1] = k;
    return [
        tensor_ops_1.tensor(allTopKVals, outputShape, xDtype),
        tensor_ops_1.tensor(allTopKIndices, outputShape, 'int32')
    ];
}
exports.topkImpl = topkImpl;
//# sourceMappingURL=topk_impl.js.map