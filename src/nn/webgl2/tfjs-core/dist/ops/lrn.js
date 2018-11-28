"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var operation_1 = require("./operation");
function localResponseNormalization_(x, depthRadius, bias, alpha, beta) {
    if (depthRadius === void 0) { depthRadius = 5; }
    if (bias === void 0) { bias = 1; }
    if (alpha === void 0) { alpha = 1; }
    if (beta === void 0) { beta = 0.5; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'localResponseNormalization');
    util.assert($x.rank === 4 || $x.rank === 3, "Error in localResponseNormalization: x must be rank 3 or 4 but got\n               rank " + $x.rank + ".");
    util.assert(util.isInt(depthRadius), "Error in localResponseNormalization: depthRadius must be an integer\n                     but got depthRadius " + depthRadius + ".");
    var x4D = $x;
    var reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
    }
    var backward = function (dy, saved) {
        var outputImage = saved[0];
        return {
            x4D: function () { return environment_1.ENV.engine.runKernel(function (backend) { return backend.LRNGrad(dy, x4D, outputImage, depthRadius, bias, alpha, beta); }, {}); }
        };
    };
    var res = environment_1.ENV.engine.runKernel(function (backend, save) { return save(backend.localResponseNormalization4D(x4D, depthRadius, bias, alpha, beta)); }, { x4D: x4D }, backward);
    if (reshapedTo4D) {
        return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
    }
    else {
        return res;
    }
}
exports.localResponseNormalization = operation_1.op({ localResponseNormalization_: localResponseNormalization_ });
//# sourceMappingURL=lrn.js.map