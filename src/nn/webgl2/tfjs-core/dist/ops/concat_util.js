"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function assertParamsConsistent(shapes, axis) {
    var rank = shapes[0].length;
    shapes.forEach(function (shape, i) {
        util.assert(shape.length === rank, "Error in concat" + rank + "D: rank of tensors[" + i + "] must be the same " +
            ("as the rank of the rest (" + rank + ")"));
    });
    util.assert(axis >= 0 && axis < rank, "Error in concat" + rank + "D: axis must be between 0 and " + (rank - 1) + ".");
    var firstShape = shapes[0];
    shapes.forEach(function (shape, i) {
        for (var r = 0; r < rank; r++) {
            util.assert((r === axis) || (shape[r] === firstShape[r]), "Error in concat" + rank + "D: Shape of tensors[" + i + "] (" + shape + ") " +
                ("does not match the shape of the rest (" + firstShape + ") ") +
                ("along the non-concatenated axis " + i + "."));
        }
    });
}
exports.assertParamsConsistent = assertParamsConsistent;
function computeOutShape(shapes, axis) {
    var outputShape = shapes[0].slice();
    for (var i = 1; i < shapes.length; i++) {
        outputShape[axis] += shapes[i][axis];
    }
    return outputShape;
}
exports.computeOutShape = computeOutShape;
//# sourceMappingURL=concat_util.js.map