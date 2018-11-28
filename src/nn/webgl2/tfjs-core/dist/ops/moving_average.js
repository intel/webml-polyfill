"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_util_1 = require("../tensor_util");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var binary_ops_1 = require("./binary_ops");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
function movingAverage_(v, x, decay, step, zeroDebias) {
    if (zeroDebias === void 0) { zeroDebias = true; }
    var $v = tensor_util_env_1.convertToTensor(v, 'v', 'movingAverage');
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'movingAverage');
    var $decay = tensor_util_env_1.convertToTensor(decay, 'decay', 'movingAverage');
    tensor_util_1.assertTypesMatch($v, $x);
    util.assert(util.arraysEqual($v.shape, $x.shape), 'Shape mismatch in v and x');
    var one = tensor_ops_1.scalar(1);
    var oneMinusDecay = one.sub($decay);
    var update = $x.sub($v).mul(oneMinusDecay);
    if (zeroDebias) {
        util.assert(step != null, 'When using zeroDebias: true, step is required.');
        var $step = tensor_util_env_1.convertToTensor(step, 'step', 'movingAverage');
        update = update.div(one.sub(binary_ops_1.pow($decay, $step)));
    }
    return $v.add(update);
}
exports.movingAverage = operation_1.op({ movingAverage_: movingAverage_ });
//# sourceMappingURL=moving_average.js.map