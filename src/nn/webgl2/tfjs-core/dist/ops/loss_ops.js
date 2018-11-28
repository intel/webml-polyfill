"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var globals_1 = require("../globals");
var tensor_util_env_1 = require("../tensor_util_env");
var util_1 = require("../util");
var axis_util_1 = require("./axis_util");
var binary_ops_1 = require("./binary_ops");
var operation_1 = require("./operation");
var tensor_ops_1 = require("./tensor_ops");
var Reduction;
(function (Reduction) {
    Reduction[Reduction["NONE"] = 0] = "NONE";
    Reduction[Reduction["MEAN"] = 1] = "MEAN";
    Reduction[Reduction["SUM"] = 2] = "SUM";
    Reduction[Reduction["SUM_BY_NONZERO_WEIGHTS"] = 3] = "SUM_BY_NONZERO_WEIGHTS";
})(Reduction = exports.Reduction || (exports.Reduction = {}));
function computeWeightedLoss_(losses, weights, reduction) {
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $losses = tensor_util_env_1.convertToTensor(losses, 'losses', 'computeWeightedLoss');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'computeWeightedLoss');
    }
    var weightedLoss = ($weights == null) ? $losses : $losses.mul($weights);
    if (reduction === Reduction.NONE) {
        return weightedLoss;
    }
    if (reduction === Reduction.SUM) {
        return weightedLoss.sum();
    }
    if (reduction === Reduction.MEAN) {
        if ($weights == null) {
            return weightedLoss.mean();
        }
        else {
            var broadcastFactor = util_1.sizeFromShape($losses.shape) / util_1.sizeFromShape($weights.shape);
            var result = weightedLoss.sum().div($weights.sum());
            return broadcastFactor > 1 ? result.div(tensor_ops_1.scalar(broadcastFactor)) :
                result;
        }
    }
    if (reduction === Reduction.SUM_BY_NONZERO_WEIGHTS) {
        if ($weights == null) {
            return weightedLoss.sum().div(tensor_ops_1.scalar($losses.size));
        }
        else {
            var broadcastedWeights = $weights.mul(tensor_ops_1.ones($losses.shape));
            var numNonZeros = broadcastedWeights.notEqual(tensor_ops_1.scalar(0)).sum().toFloat();
            return weightedLoss.sum().div(numNonZeros);
        }
    }
    throw Error("Unknown reduction: " + reduction);
}
function absoluteDifference_(labels, predictions, weights, reduction) {
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'absoluteDifference');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'absoluteDifference');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'absoluteDifference');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in absoluteDifference: ');
    var losses = $labels.sub($predictions).abs();
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function meanSquaredError_(labels, predictions, weights, reduction) {
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'meanSquaredError');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'meanSquaredError');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'meanSquaredError');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in meanSquaredError: ');
    var losses = $labels.squaredDifference($predictions);
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function cosineDistance_(labels, predictions, axis, weights, reduction) {
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'cosineDistance');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'cosineDistance');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'cosineDistance');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in cosineDistance: ');
    var one = tensor_ops_1.scalar(1);
    var losses = one.sub($labels.mul($predictions).sum(axis, true));
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function hingeLoss_(labels, predictions, weights, reduction) {
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'hingeLoss');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'hingeLoss');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'hingeLoss');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in hingeLoss: ');
    var one = tensor_ops_1.scalar(1);
    $labels = tensor_ops_1.scalar(2).mul($labels).sub(one);
    var losses = one.sub($labels.mul($predictions)).relu();
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function logLoss_(labels, predictions, weights, epsilon, reduction) {
    if (epsilon === void 0) { epsilon = 1e-7; }
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'logLoss');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'logLoss');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'logLoss');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in logLoss: ');
    var one = tensor_ops_1.scalar(1);
    var epsilonScalar = tensor_ops_1.scalar(epsilon);
    var losses = $labels.mul($predictions.add(epsilonScalar).log())
        .neg()
        .sub(one.sub($labels).mul(one.sub($predictions).add(epsilonScalar).log()));
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function sigmoidCrossEntropyWithLogits_(labels, logits) {
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'sigmoidCrossEntropyWithLogits');
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'sigmoidCrossEntropyWithLogits');
    util_1.assertShapesMatch($labels.shape, $logits.shape, 'Error in sigmoidCrossEntropyWithLogits: ');
    var maxOutput = $logits.relu();
    var outputXTarget = $logits.mul($labels);
    var sigmoidOutput = $logits.abs().neg().exp().log1p();
    return maxOutput.sub(outputXTarget).add(sigmoidOutput);
}
function sigmoidCrossEntropy_(multiClassLabels, logits, weights, labelSmoothing, reduction) {
    if (labelSmoothing === void 0) { labelSmoothing = 0; }
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $multiClassLabels = tensor_util_env_1.convertToTensor(multiClassLabels, 'multiClassLabels', 'sigmoidCrossEntropy');
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'sigmoidCrossEntropy');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'sigmoidCrossEntropy');
    }
    util_1.assertShapesMatch($multiClassLabels.shape, $logits.shape, 'Error in sigmoidCrossEntropy: ');
    if (labelSmoothing > 0) {
        var labelSmoothingScalar = tensor_ops_1.scalar(labelSmoothing);
        var one = tensor_ops_1.scalar(1);
        var half = tensor_ops_1.scalar(0.5);
        $multiClassLabels = $multiClassLabels.mul(one.sub(labelSmoothingScalar))
            .add(half.mul(labelSmoothingScalar));
    }
    var losses = sigmoidCrossEntropyWithLogits_($multiClassLabels, $logits);
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function huberLoss_(labels, predictions, weights, delta, reduction) {
    if (delta === void 0) { delta = 1.0; }
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $labels = tensor_util_env_1.convertToTensor(labels, 'labels', 'huberLoss');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'predictions', 'huberLoss');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'huberLoss');
    }
    util_1.assertShapesMatch($labels.shape, $predictions.shape, 'Error in huberLoss: ');
    var deltaScalar = tensor_ops_1.scalar(delta);
    var error = $predictions.sub($labels).abs();
    var quadratic = binary_ops_1.minimum(error, deltaScalar);
    var linear = error.sub(quadratic);
    var losses = tensor_ops_1.scalar(0.5).mul(quadratic.square()).add(deltaScalar.mul(linear));
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
function softmaxCrossEntropyWithLogits_(labels, logits, dim) {
    if (dim === void 0) { dim = -1; }
    if (dim === -1) {
        dim = logits.rank - 1;
    }
    if (dim !== logits.rank - 1) {
        throw Error("Softmax cross entropy along a non-last dimension is not yet " +
            ("supported. Labels / logits was rank " + logits.rank + " ") +
            ("and dim was " + dim));
    }
    var customOp = globals_1.customGrad(function (labels, logits) {
        var keepDims = true;
        var lse = logits.logSumExp([dim], keepDims);
        var logResult = logits.toFloat().sub(lse);
        var costVector = logResult.mul(labels).neg();
        var value = costVector.sum([dim]);
        var gradFunc = function (dy) {
            var dyShape = axis_util_1.expandShapeToKeepDim(dy.shape, [dim]);
            return [
                dy.reshape(dyShape).mul(labels.toFloat().sub(logResult.exp())),
                dy.reshape(dyShape).mul(logResult.exp().sub(labels.toFloat())),
            ];
        };
        return { value: value, gradFunc: gradFunc };
    });
    return customOp(labels, logits);
}
function softmaxCrossEntropy_(onehotLabels, logits, weights, labelSmoothing, reduction) {
    if (labelSmoothing === void 0) { labelSmoothing = 0; }
    if (reduction === void 0) { reduction = Reduction.SUM_BY_NONZERO_WEIGHTS; }
    var $onehotLabels = tensor_util_env_1.convertToTensor(onehotLabels, 'onehotLabels', 'softmaxCrossEntropy');
    var $logits = tensor_util_env_1.convertToTensor(logits, 'logits', 'softmaxCrossEntropy');
    var $weights = null;
    if (weights != null) {
        $weights = tensor_util_env_1.convertToTensor(weights, 'weights', 'softmaxCrossEntropy');
    }
    util_1.assertShapesMatch($onehotLabels.shape, $logits.shape, 'Error in softmaxCrossEntropy: ');
    if (labelSmoothing > 0) {
        var labelSmoothingScalar = tensor_ops_1.scalar(labelSmoothing);
        var one = tensor_ops_1.scalar(1);
        var numClasses = tensor_ops_1.scalar($onehotLabels.shape[1]);
        $onehotLabels = $onehotLabels.mul(one.sub(labelSmoothingScalar))
            .add(labelSmoothingScalar.div(numClasses));
    }
    var losses = softmaxCrossEntropyWithLogits_($onehotLabels, $logits);
    return exports.computeWeightedLoss(losses, $weights, reduction);
}
exports.absoluteDifference = operation_1.op({ absoluteDifference_: absoluteDifference_ });
exports.computeWeightedLoss = operation_1.op({ computeWeightedLoss_: computeWeightedLoss_ });
exports.cosineDistance = operation_1.op({ cosineDistance_: cosineDistance_ });
exports.hingeLoss = operation_1.op({ hingeLoss_: hingeLoss_ });
exports.huberLoss = operation_1.op({ huberLoss_: huberLoss_ });
exports.logLoss = operation_1.op({ logLoss_: logLoss_ });
exports.meanSquaredError = operation_1.op({ meanSquaredError_: meanSquaredError_ });
exports.sigmoidCrossEntropy = operation_1.op({ sigmoidCrossEntropy_: sigmoidCrossEntropy_ });
exports.softmaxCrossEntropy = operation_1.op({ softmaxCrossEntropy_: softmaxCrossEntropy_ });
//# sourceMappingURL=loss_ops.js.map