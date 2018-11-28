"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var array_ops_1 = require("../ops/array_ops");
function whereImpl(condShape, condVals) {
    var indices = [];
    for (var i = 0; i < condVals.length; i++) {
        if (condVals[i]) {
            indices.push(i);
        }
    }
    var inBuffer = array_ops_1.buffer(condShape, 'int32');
    var out = array_ops_1.buffer([indices.length, condShape.length], 'int32');
    for (var i = 0; i < indices.length; i++) {
        var loc = inBuffer.indexToLoc(indices[i]);
        var offset = i * condShape.length;
        out.values.set(loc, offset);
    }
    return out.toTensor();
}
exports.whereImpl = whereImpl;
//# sourceMappingURL=where_impl.js.map