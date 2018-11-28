"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_1 = require("./tensor");
var util_1 = require("./util");
function assertTypesMatch(a, b) {
    util_1.assert(a.dtype === b.dtype, "The dtypes of the first(" + a.dtype + ") and" +
        (" second(" + b.dtype + ") input must match"));
}
exports.assertTypesMatch = assertTypesMatch;
function isTensorInList(tensor, tensorList) {
    for (var i = 0; i < tensorList.length; i++) {
        if (tensorList[i].id === tensor.id) {
            return true;
        }
    }
    return false;
}
exports.isTensorInList = isTensorInList;
function flattenNameArrayMap(nameArrayMap, keys) {
    var xs = [];
    if (nameArrayMap instanceof tensor_1.Tensor) {
        xs.push(nameArrayMap);
    }
    else {
        var xMap = nameArrayMap;
        for (var i = 0; i < keys.length; i++) {
            xs.push(xMap[keys[i]]);
        }
    }
    return xs;
}
exports.flattenNameArrayMap = flattenNameArrayMap;
function unflattenToNameArrayMap(keys, flatArrays) {
    if (keys.length !== flatArrays.length) {
        throw new Error("Cannot unflatten Tensor[], keys and arrays are not of same length.");
    }
    var result = {};
    for (var i = 0; i < keys.length; i++) {
        result[keys[i]] = flatArrays[i];
    }
    return result;
}
exports.unflattenToNameArrayMap = unflattenToNameArrayMap;
function getTensorsInContainer(result) {
    var list = [];
    var seen = new Set();
    walkTensorContainer(result, list, seen);
    return list;
}
exports.getTensorsInContainer = getTensorsInContainer;
function walkTensorContainer(container, list, seen) {
    if (container == null) {
        return;
    }
    if (container instanceof tensor_1.Tensor) {
        list.push(container);
        return;
    }
    if (!isIterable(container)) {
        return;
    }
    var iterable = container;
    for (var k in iterable) {
        var val = iterable[k];
        if (!seen.has(val)) {
            seen.add(val);
            walkTensorContainer(val, list, seen);
        }
    }
}
function isIterable(obj) {
    return Array.isArray(obj) || typeof obj === 'object';
}
//# sourceMappingURL=tensor_util.js.map