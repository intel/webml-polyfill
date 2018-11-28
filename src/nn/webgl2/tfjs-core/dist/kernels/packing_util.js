"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getVecChannels(name, rank) {
    return ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank).map(function (d) { return name + "." + d; });
}
exports.getVecChannels = getVecChannels;
function getChannels(name, rank) {
    if (rank === 1) {
        return [name];
    }
    return getVecChannels(name, rank);
}
exports.getChannels = getChannels;
function getSourceCoords(rank, dims) {
    if (rank === 1) {
        return 'rc';
    }
    var coords = '';
    for (var i = 0; i < rank; i++) {
        coords += dims[i];
        if (i < rank - 1) {
            coords += ',';
        }
    }
    return coords;
}
exports.getSourceCoords = getSourceCoords;
//# sourceMappingURL=packing_util.js.map