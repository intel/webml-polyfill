"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var array_ops_1 = require("./array_ops");
var broadcast_util_1 = require("./broadcast_util");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
var unary_ops_1 = require("./unary_ops");
function batchNormalization2d_(x, mean, variance, varianceEpsilon, scale, offset) {
    if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'batchNormalization');
    var $mean = tensor_util_env_1.convertToTensor(mean, 'mean', 'batchNormalization');
    var $variance = tensor_util_env_1.convertToTensor(variance, 'variance', 'batchNormalization');
    var $scale;
    if (scale != null) {
        $scale = tensor_util_env_1.convertToTensor(scale, 'scale', 'batchNormalization');
    }
    var $offset;
    if (offset != null) {
        $offset = tensor_util_env_1.convertToTensor(offset, 'offset', 'batchNormalization');
    }
    util.assert($x.rank === 2, "Error in batchNormalization3D: x must be rank 3 but got rank " +
        ($x.rank + "."));
    util.assert($mean.rank === 2 || $mean.rank === 1, "Error in batchNormalization2D: mean must be rank 2 or rank 1 but " +
        ("got rank " + $mean.rank + "."));
    util.assert($variance.rank === 2 || $variance.rank === 1, "Error in batchNormalization2D: variance must be rank 2 or rank 1 " +
        ("but got rank " + $variance.rank + "."));
    if ($scale != null) {
        util.assert($scale.rank === 2 || $scale.rank === 1, "Error in batchNormalization2D: scale must be rank 2 or rank 1 " +
            ("but got rank " + $scale.rank + "."));
    }
    if ($offset != null) {
        util.assert($offset.rank === 2 || $offset.rank === 1, "Error in batchNormalization2D: offset must be rank 2 or rank 1 " +
            ("but got rank " + $offset.rank + "."));
    }
    return exports.batchNormalization($x, $mean, $variance, varianceEpsilon, $scale, $offset);
}
function batchNormalization3d_(x, mean, variance, varianceEpsilon, scale, offset) {
    if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'batchNormalization');
    var $mean = tensor_util_env_1.convertToTensor(mean, 'mean', 'batchNormalization');
    var $variance = tensor_util_env_1.convertToTensor(variance, 'variance', 'batchNormalization');
    var $scale;
    if (scale != null) {
        $scale = tensor_util_env_1.convertToTensor(scale, 'scale', 'batchNormalization');
    }
    var $offset;
    if (offset != null) {
        $offset = tensor_util_env_1.convertToTensor(offset, 'offset', 'batchNormalization');
    }
    util.assert($x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
        ($x.rank + "."));
    util.assert($mean.rank === 3 || $mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
        ("got rank " + $mean.rank + "."));
    util.assert($variance.rank === 3 || $variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
        ("but got rank " + $variance.rank + "."));
    if ($scale != null) {
        util.assert($scale.rank === 3 || $scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
            ("but got rank " + $scale.rank + "."));
    }
    if ($offset != null) {
        util.assert($offset.rank === 3 || $offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
            ("but got rank " + $offset.rank + "."));
    }
    return exports.batchNormalization($x, $mean, $variance, varianceEpsilon, $scale, $offset);
}
function batchNormalization4d_(x, mean, variance, varianceEpsilon, scale, offset) {
    if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'batchNormalization');
    var $mean = tensor_util_env_1.convertToTensor(mean, 'mean', 'batchNormalization');
    var $variance = tensor_util_env_1.convertToTensor(variance, 'variance', 'batchNormalization');
    var $scale;
    if (scale != null) {
        $scale = tensor_util_env_1.convertToTensor(scale, 'scale', 'batchNormalization');
    }
    var $offset;
    if (offset != null) {
        $offset = tensor_util_env_1.convertToTensor(offset, 'offset', 'batchNormalization');
    }
    util.assert($x.rank === 4, "Error in batchNormalization4D: x must be rank 4 but got rank " +
        ($x.rank + "."));
    util.assert($mean.rank === 4 || $mean.rank === 1, "Error in batchNormalization4D: mean must be rank 4 or rank 1 but " +
        ("got rank " + $mean.rank + "."));
    util.assert($variance.rank === 4 || $variance.rank === 1, "Error in batchNormalization4D: variance must be rank 4 or rank 1 " +
        ("but got rank " + $variance.rank + "."));
    if ($scale != null) {
        util.assert($scale.rank === 4 || $scale.rank === 1, "Error in batchNormalization4D: scale must be rank 4 or rank 1 " +
            ("but got rank " + $scale.rank + "."));
    }
    if ($offset != null) {
        util.assert($offset.rank === 4 || $offset.rank === 1, "Error in batchNormalization4D: offset must be rank 4 or rank 1 " +
            ("but got rank " + $offset.rank + "."));
    }
    return exports.batchNormalization($x, $mean, $variance, varianceEpsilon, $scale, $offset);
}
function batchNormalization_(x, mean, variance, varianceEpsilon, scale, offset) {
    if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'batchNormalization');
    var $mean = tensor_util_env_1.convertToTensor(mean, 'mean', 'batchNormalization');
    var $variance = tensor_util_env_1.convertToTensor(variance, 'variance', 'batchNormalization');
    var $scale;
    if (scale != null) {
        $scale = tensor_util_env_1.convertToTensor(scale, 'scale', 'batchNormalization');
    }
    var $offset;
    if (offset != null) {
        $offset = tensor_util_env_1.convertToTensor(offset, 'offset', 'batchNormalization');
    }
    util.assert($mean.rank === $variance.rank, 'Batch normalization gradient requires mean and variance to have ' +
        'equal ranks.');
    util.assert($offset == null || $mean.rank === $offset.rank, 'Batch normalization gradient requires mean and offset to have ' +
        'equal ranks.');
    util.assert($scale == null || $mean.rank === $scale.rank, 'Batch normalization gradient requires mean and scale to have ' +
        'equal ranks.');
    var x4D;
    if ($x.rank === 0 || $x.rank === 1) {
        x4D = $x.as4D(1, 1, 1, $x.size);
    }
    else if ($x.rank === 2) {
        x4D = $x.as4D(1, 1, $x.shape[0], $x.shape[1]);
    }
    else if ($x.rank === 3) {
        x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
    }
    else {
        x4D = $x;
    }
    var der = function (dy) {
        var scaleValue = $scale == null ? tensor_ops_1.scalar(1) : $scale;
        var reductionAxes = broadcast_util_1.getReductionAxes($mean.shape, x4D.shape);
        var tileShape = [];
        if ($mean.rank === 1) {
            for (var i = 0; i < x4D.shape.length - 1; ++i) {
                tileShape.push(x4D.shape[i]);
            }
            tileShape.push(1);
        }
        var xMinusMean = $x.sub($mean);
        var dyTimesScaleValue = dy.mul(scaleValue);
        var oneOverSqrtVariance = unary_ops_1.rsqrt($variance.add(tensor_ops_1.scalar(varianceEpsilon)));
        var minusHalfRCube = oneOverSqrtVariance.mul(oneOverSqrtVariance)
            .mul(oneOverSqrtVariance)
            .mul(tensor_ops_1.scalar(-0.5));
        var derX = function () {
            if ($mean.rank === 1) {
                return dy
                    .mul(array_ops_1.tile(oneOverSqrtVariance.as4D(1, 1, 1, $mean.shape[0]), tileShape))
                    .mul(scaleValue)
                    .reshape($x.shape);
            }
            else {
                return dy.mul(oneOverSqrtVariance).mul(scaleValue).reshape($x.shape);
            }
        };
        var derMean = function () {
            var meanDer = oneOverSqrtVariance.mul(tensor_ops_1.scalar(-1)).mul(dyTimesScaleValue);
            if ($mean.rank === 1) {
                meanDer = meanDer.sum(reductionAxes);
            }
            return meanDer.reshape($mean.shape);
        };
        var derVariance = function () {
            var varianceDer = minusHalfRCube.mul(xMinusMean).mul(dyTimesScaleValue);
            if ($mean.rank === 1) {
                varianceDer = varianceDer.sum(reductionAxes);
            }
            return varianceDer.reshape($mean.shape);
        };
        var derScale = function () {
            var xMinusMean2TimesRsqrt = xMinusMean.mul(oneOverSqrtVariance);
            var scaleDer = dy.mul(xMinusMean2TimesRsqrt);
            if ($mean.rank === 1) {
                scaleDer = scaleDer.sum(reductionAxes);
            }
            return scaleDer.reshape($mean.shape);
        };
        var derOffset = function () {
            var offsetDer = dy;
            if ($mean.rank === 1) {
                offsetDer = offsetDer.sum(reductionAxes);
            }
            return offsetDer.reshape($mean.shape);
        };
        return {
            $x: derX,
            $mean: derMean,
            $variance: derVariance,
            $scale: derScale,
            $offset: derOffset
        };
    };
    var res = environment_1.ENV.engine.runKernel(function (backend) { return backend.batchNormalization(x4D, batchnormReshape4D($mean), batchnormReshape4D($variance), varianceEpsilon, batchnormReshape4D($scale), batchnormReshape4D($offset)); }, { $x: $x, $mean: $mean, $variance: $variance, $scale: $scale, $offset: $offset }, der);
    return res.reshape($x.shape);
}
function batchnormReshape4D(x) {
    if (x == null) {
        return null;
    }
    if (x.rank === 0) {
        return x.as1D();
    }
    else if (x.rank === 1) {
        return x;
    }
    else if (x.rank === 2) {
        return x.as4D(1, 1, x.shape[0], x.shape[1]);
    }
    else if (x.rank === 3) {
        return x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }
    return x;
}
exports.batchNormalization2d = operation_1.op({ batchNormalization2d_: batchNormalization2d_ });
exports.batchNormalization3d = operation_1.op({ batchNormalization3d_: batchNormalization3d_ });
exports.batchNormalization4d = operation_1.op({ batchNormalization4d_: batchNormalization4d_ });
exports.batchNormalization = operation_1.op({ batchNormalization_: batchNormalization_ });
//# sourceMappingURL=batchnorm.js.map