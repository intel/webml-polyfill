"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var operation_1 = require("./operation");
var slice_util = require("./slice_util");
function slice1d_(x, begin, size) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'slice1d');
    util.assert($x.rank === 1, "slice1d expects a rank-1 tensor, but got a rank-" + $x.rank + " tensor");
    return exports.slice($x, [begin], [size]);
}
function slice2d_(x, begin, size) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'slice2d');
    util.assert($x.rank === 2, "slice2d expects a rank-2 tensor, but got a rank-" + $x.rank + " tensor");
    return exports.slice($x, begin, size);
}
function slice3d_(x, begin, size) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'slice3d');
    util.assert($x.rank === 3, "slice3d expects a rank-3 tensor, but got a rank-" + $x.rank + " tensor");
    return exports.slice($x, begin, size);
}
function slice4d_(x, begin, size) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'slice4d');
    util.assert($x.rank === 4, "slice4d expects a rank-4 tensor, but got a rank-" + $x.rank + " tensor");
    return exports.slice($x, begin, size);
}
function slice_(x, begin, size) {
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'slice');
    if ($x.rank === 0) {
        throw new Error('Slicing scalar is not possible');
    }
    var begin_;
    if (typeof begin === 'number') {
        begin_ = [begin].concat(new Array($x.rank - 1).fill(0));
    }
    else if (begin.length < $x.rank) {
        begin_ = begin.concat(new Array($x.rank - begin.length).fill(0));
    }
    else {
        begin_ = begin.slice();
    }
    var size_;
    if (size == null) {
        size_ = new Array($x.rank).fill(-1);
    }
    else if (typeof size === 'number') {
        size_ = [size].concat(new Array($x.rank - 1).fill(-1));
    }
    else if (size.length < $x.rank) {
        size_ = size.concat(new Array($x.rank - size.length).fill(-1));
    }
    else {
        size_ = size;
    }
    size_ = size_.map(function (d, i) {
        if (d >= 0) {
            return d;
        }
        else {
            util.assert(d === -1, 'Bad value in size');
            return $x.shape[i] - begin_[i];
        }
    });
    slice_util.assertParamsValid($x, begin_, size_);
    var inputShape = $x.shape;
    var grad = function (dy) {
        var paddings = [];
        for (var i = 0; i < dy.rank; i++) {
            paddings.push([begin_[i], inputShape[i] - begin_[i] - size_[i]]);
        }
        return { $x: function () { return dy.pad(paddings); } };
    };
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.slice($x, begin_, size_); }, { $x: $x }, grad);
}
exports.slice = operation_1.op({ slice_: slice_ });
exports.slice1d = operation_1.op({ slice1d_: slice1d_ });
exports.slice2d = operation_1.op({ slice2d_: slice2d_ });
exports.slice3d = operation_1.op({ slice3d_: slice3d_ });
exports.slice4d = operation_1.op({ slice4d_: slice4d_ });
//# sourceMappingURL=slice.js.map