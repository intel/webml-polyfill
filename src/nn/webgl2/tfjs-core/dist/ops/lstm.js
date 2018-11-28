"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
function multiRNNCell_(lstmCells, data, c, h) {
    var $data = tensor_util_env_1.convertToTensor(data, 'data', 'multiRNNCell');
    var $c = tensor_util_env_1.convertToTensorArray(c, 'c', 'multiRNNCell');
    var $h = tensor_util_env_1.convertToTensorArray(h, 'h', 'multiRNNCell');
    var input = $data;
    var newStates = [];
    for (var i = 0; i < lstmCells.length; i++) {
        var output = lstmCells[i](input, $c[i], $h[i]);
        newStates.push(output[0]);
        newStates.push(output[1]);
        input = output[1];
    }
    var newC = [];
    var newH = [];
    for (var i = 0; i < newStates.length; i += 2) {
        newC.push(newStates[i]);
        newH.push(newStates[i + 1]);
    }
    return [newC, newH];
}
function basicLSTMCell_(forgetBias, lstmKernel, lstmBias, data, c, h) {
    var $forgetBias = tensor_util_env_1.convertToTensor(forgetBias, 'forgetBias', 'basicLSTMCell');
    var $lstmKernel = tensor_util_env_1.convertToTensor(lstmKernel, 'lstmKernel', 'basicLSTMCell');
    var $lstmBias = tensor_util_env_1.convertToTensor(lstmBias, 'lstmBias', 'basicLSTMCell');
    var $data = tensor_util_env_1.convertToTensor(data, 'data', 'basicLSTMCell');
    var $c = tensor_util_env_1.convertToTensor(c, 'c', 'basicLSTMCell');
    var $h = tensor_util_env_1.convertToTensor(h, 'h', 'basicLSTMCell');
    var combined = $data.concat($h, 1);
    var weighted = combined.matMul($lstmKernel);
    var res = weighted.add($lstmBias);
    var batchSize = res.shape[0];
    var sliceCols = res.shape[1] / 4;
    var sliceSize = [batchSize, sliceCols];
    var i = res.slice([0, 0], sliceSize);
    var j = res.slice([0, sliceCols], sliceSize);
    var f = res.slice([0, sliceCols * 2], sliceSize);
    var o = res.slice([0, sliceCols * 3], sliceSize);
    var newC = i.sigmoid().mulStrict(j.tanh()).addStrict($c.mulStrict($forgetBias.add(f).sigmoid()));
    var newH = newC.tanh().mulStrict(o.sigmoid());
    return [newC, newH];
}
exports.basicLSTMCell = operation_1.op({ basicLSTMCell_: basicLSTMCell_ });
exports.multiRNNCell = operation_1.op({ multiRNNCell_: multiRNNCell_ });
//# sourceMappingURL=lstm.js.map