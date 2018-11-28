"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var axis_util = require("./axis_util");
var operation_1 = require("./operation");
function transpose_(x, perm) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'transpose');
    if (perm == null) {
        perm = $x.shape.map(function (s, i) { return i; }).reverse();
    }
    util.assert($x.rank === perm.length, "Error in transpose: rank of input " + $x.rank + " " +
        ("must match length of perm " + perm + "."));
    perm.forEach(function (axis) {
        util.assert(axis >= 0 && axis < $x.rank, "All entries in 'perm' must be between 0 and " + ($x.rank - 1) +
            (" but got " + perm));
    });
    if ($x.rank <= 1) {
        return $x.clone();
    }
    var der = function (dy) {
        var undoPerm = axis_util.getUndoAxesPermutation(perm);
        return { $x: function () { return dy.transpose(undoPerm); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.transpose($x, perm); }, { $x: $x }, der);
}
exports.transpose = operation_1.op({ transpose_: transpose_ });
//# sourceMappingURL=transpose.js.map