"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function split(x, sizeSplits, axis) {
    var begin = Array(x.rank).fill(0);
    var size = x.shape.slice();
    return sizeSplits.map(function (s) {
        size[axis] = s;
        var slice = x.slice(begin, size);
        begin[axis] += s;
        return slice;
    });
}
exports.split = split;
//# sourceMappingURL=split_shared.js.map