"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
function topk_(x, k, sorted) {
    if (k === void 0) { k = 1; }
    if (sorted === void 0) { sorted = true; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'topk');
    if ($x.rank === 0) {
        throw new Error('topk() expects the input to be of rank 1 or higher');
    }
    var lastDim = $x.shape[$x.shape.length - 1];
    if (k > lastDim) {
        throw new Error("'k' passed to topk() must be <= the last dimension (" + lastDim + ") " +
            ("but got " + k));
    }
    var _a = environment_1.ENV.engine.runKernel(function (b) { return b.topk($x, k, sorted); }, { $x: $x }), values = _a[0], indices = _a[1];
    return { values: values, indices: indices };
}
exports.topk = operation_1.op({ topk_: topk_ });
//# sourceMappingURL=topk.js.map