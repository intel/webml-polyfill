"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function getLogicalCoordinatesFromFlatIndex(coords, shape, index) {
    if (index === void 0) { index = 'index'; }
    var strides = util.computeStrides(shape);
    return strides
        .map(function (stride, i) {
        var line1 = "int " + coords[i] + " = " + index + " / " + stride;
        var line2 = i === strides.length - 1 ?
            "int " + coords[i + 1] + " = " + index + " - " + coords[i] + " * " + stride :
            "index -= " + coords[i] + " * " + stride;
        return line1 + "; " + line2 + ";";
    })
        .join('');
}
exports.getLogicalCoordinatesFromFlatIndex = getLogicalCoordinatesFromFlatIndex;
function buildVec(x) {
    if (x.length === 1) {
        return "" + x[0];
    }
    return "vec" + x.length + "(" + x.join(',') + ")";
}
function dotify(x, y) {
    if (x.length !== y.length) {
        throw new Error("Vectors to be dotted must be of the same length -" +
            ("got " + x.length + " and " + y.length));
    }
    var slices = [];
    var nearestVec4 = Math.floor(x.length / 4);
    var nearestVec4Remainder = x.length % 4;
    for (var i = 0; i < nearestVec4; i++) {
        var xSlice = x.slice(i * 4, i * 4 + 4);
        var ySlice = y.slice(i * 4, i * 4 + 4);
        slices.push(buildVec(xSlice) + ", " + buildVec(ySlice));
    }
    if (nearestVec4Remainder !== 0) {
        var xSlice = x.slice(nearestVec4 * 4);
        var ySlice = y.slice(nearestVec4 * 4);
        if (xSlice.length === 1) {
            xSlice = xSlice.map(function (d) { return "float(" + d + ")"; });
            ySlice = ySlice.map(function (d) { return "float(" + d + ")"; });
        }
        slices.push(buildVec(xSlice) + ", " + buildVec(ySlice));
    }
    return slices.map(function (d, i) { return "dot(" + d + ")"; }).join('+');
}
exports.dotify = dotify;
//# sourceMappingURL=shader_compiler_util.js.map