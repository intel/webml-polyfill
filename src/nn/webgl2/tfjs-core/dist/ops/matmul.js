"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var operation_1 = require("./operation");
function matMul_(a, b, transposeA, transposeB) {
    if (transposeA === void 0) { transposeA = false; }
    if (transposeB === void 0) { transposeB = false; }
    var $a = tensor_util_env_1.convertToTensor(a, 'a', 'matMul');
    var $b = tensor_util_env_1.convertToTensor(b, 'b', 'matMul');
    var innerShapeA = transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
    var innerShapeB = transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];
    var outerShapeA = transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
    var outerShapeB = transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];
    var outerDimsA = $a.shape.slice(0, -2);
    var outerDimsB = $b.shape.slice(0, -2);
    var batchDimA = util.sizeFromShape(outerDimsA);
    var batchDimB = util.sizeFromShape(outerDimsB);
    util.assert($a.rank >= 2 && $b.rank >= 2 && $a.rank === $b.rank, "Error in matMul: inputs must have the same rank of at least 2, " +
        ("got ranks " + $a.rank + " and " + $b.rank + "."));
    util.assert(util.arraysEqual(outerDimsA, outerDimsB), "Error in matMul: outer dimensions (" + outerDimsA + ") and (" +
        (outerDimsB + ") of Tensors with shapes " + $a.shape + " and ") +
        ($b.shape + " must match."));
    util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
        (innerShapeB + ") of Tensors with shapes " + $a.shape + " and ") +
        ($b.shape + " and transposeA=" + transposeA) +
        (" and transposeB=" + transposeB + " must match."));
    var outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);
    var a3D = transposeA ? $a.as3D(batchDimA, innerShapeA, outerShapeA) :
        $a.as3D(batchDimA, outerShapeA, innerShapeA);
    var b3D = transposeB ? $b.as3D(batchDimB, outerShapeB, innerShapeB) :
        $b.as3D(batchDimB, innerShapeB, outerShapeB);
    var grad = function (dy) {
        if (!transposeA && !transposeB) {
            return {
                $a: function () { return dy.matMul(b3D.toFloat(), false, true); },
                $b: function () { return a3D.toFloat().matMul(dy, true, false); }
            };
        }
        else if (!transposeA && transposeB) {
            return {
                $a: function () { return dy.matMul(b3D.toFloat(), false, false); },
                $b: function () { return dy.matMul(a3D.toFloat(), true, false); }
            };
        }
        else if (transposeA && !transposeB) {
            return {
                $a: function () { return b3D.toFloat().matMul(dy, false, true); },
                $b: function () { return a3D.toFloat().matMul(dy, false, false); }
            };
        }
        else {
            return {
                $a: function () { return b3D.toFloat().matMul(dy, true, true); },
                $b: function () { return dy.matMul(a3D.toFloat(), true, true); }
            };
        }
    };
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.batchMatMul(a3D, b3D, transposeA, transposeB); }, { $a: a3D, $b: b3D }, grad);
    return res.reshape(outShape);
}
function outerProduct_(v1, v2) {
    var $v1 = tensor_util_env_1.convertToTensor(v1, 'v1', 'outerProduct');
    var $v2 = tensor_util_env_1.convertToTensor(v2, 'v2', 'outerProduct');
    util.assert($v1.rank === 1 && $v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
        ($v1.rank + " and " + $v2.rank + "."));
    return $v1.as2D(-1, 1).matMul($v2.as2D(1, -1));
}
function dot_(t1, t2) {
    var $t1 = tensor_util_env_1.convertToTensor(t1, 't1', 'dot');
    var $t2 = tensor_util_env_1.convertToTensor(t2, 't2', 'dot');
    util.assert(($t1.rank === 1 || $t1.rank === 2) && ($t2.rank === 1 || $t2.rank === 2), "Error in dot: inputs must all be rank 1 or 2, but got ranks " +
        ($t1.rank + " and " + $t2.rank + "."));
    var t1Inner = ($t1.rank === 1 ? $t1.size : $t1.shape[1]);
    var t2Inner = ($t2.rank === 1 ? $t2.size : $t2.shape[0]);
    util.assert(t1Inner === t2Inner, "Error in dot: inner dimensions of inputs must match, but got " +
        (t1Inner + " and " + t2Inner + "."));
    if ($t1.rank === 1 && $t2.rank === 1) {
        return $t1.as2D(1, -1).matMul($t2.as2D(-1, 1)).asScalar();
    }
    else if ($t1.rank === 1 && $t2.rank === 2) {
        return $t1.as2D(1, -1).matMul($t2.as2D($t2.shape[0], $t2.shape[1])).as1D();
    }
    else if ($t1.rank === 2 && $t2.rank === 1) {
        return $t1.matMul($t2.as2D(-1, 1)).as1D();
    }
    else {
        return $t1.matMul($t2.as2D($t2.shape[0], $t2.shape[1]));
    }
}
exports.matMul = operation_1.op({ matMul_: matMul_ });
exports.dot = operation_1.op({ dot_: dot_ });
exports.outerProduct = operation_1.op({ outerProduct_: outerProduct_ });
//# sourceMappingURL=matmul.js.map