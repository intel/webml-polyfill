"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var array_ops_1 = require("./array_ops");
var operation_1 = require("./operation");
function confusionMatrix_(labels, predictions, numClasses) {
    var $labels = tensor_util_env_1.convertToTensor(labels, 'label', 'confusionMatrix', 'int32');
    var $predictions = tensor_util_env_1.convertToTensor(predictions, 'label', 'confusionMatrix', 'int32');
    util.assert(numClasses == null || numClasses > 0 && Number.isInteger(numClasses), "If provided, numClasses must be a positive integer, " +
        ("but got " + numClasses));
    util.assert($labels.rank === 1, "Expected the rank of labels to be 1, but got " + $labels.rank);
    util.assert($predictions.rank === 1, "Expected the rank of predictions to be 1, " +
        ("but got " + $predictions.rank));
    util.assert($labels.shape[0] === $predictions.shape[0], "Mismatch in the number of examples: " +
        ($labels.shape[0] + " vs. " + $predictions.shape[0] + ". ") +
        "Labels and predictions should have the same number of elements.");
    util.assert(numClasses > 0 && Number.isInteger(numClasses), "numClasses is required to be a positive integer, but got " + numClasses);
    var oneHotLabels = array_ops_1.oneHot($labels.asType('int32'), numClasses);
    var oneHotPredictions = array_ops_1.oneHot($predictions.asType('int32'), numClasses);
    return oneHotLabels.transpose().matMul(oneHotPredictions).asType('int32');
}
exports.confusionMatrix_ = confusionMatrix_;
exports.confusionMatrix = operation_1.op({ confusionMatrix_: confusionMatrix_ });
//# sourceMappingURL=confusion_matrix.js.map