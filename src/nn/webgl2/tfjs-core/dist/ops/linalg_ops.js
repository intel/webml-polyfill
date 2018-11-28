"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var globals_1 = require("../globals");
var util_1 = require("../util");
var array_ops_1 = require("./array_ops");
var concat_split_1 = require("./concat_split");
var norm_1 = require("./norm");
var operation_1 = require("./operation");
var reduction_ops_1 = require("./reduction_ops");
var tensor_ops_1 = require("./tensor_ops");
function gramSchmidt_(xs) {
    var inputIsTensor2D;
    if (Array.isArray(xs)) {
        inputIsTensor2D = false;
        util_1.assert(xs != null && xs.length > 0, 'Gram-Schmidt process: input must not be null, undefined, or empty');
        var dim = xs[0].shape[0];
        for (var i = 1; i < xs.length; ++i) {
            util_1.assert(xs[i].shape[0] === dim, 'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
                ("(" + xs[i].shape[0] + " vs. " + dim + ")"));
        }
    }
    else {
        inputIsTensor2D = true;
        xs = concat_split_1.split(xs, xs.shape[0], 0).map(function (x) { return array_ops_1.squeeze(x, [0]); });
    }
    util_1.assert(xs.length <= xs[0].shape[0], "Gram-Schmidt: Number of vectors (" + xs.length + ") exceeds " +
        ("number of dimensions (" + xs[0].shape[0] + ")."));
    var ys = [];
    var xs1d = xs;
    var _loop_1 = function (i) {
        ys.push(environment_1.ENV.engine.tidy(function () {
            var x = xs1d[i];
            if (i > 0) {
                for (var j = 0; j < i; ++j) {
                    var proj = reduction_ops_1.sum(ys[j].mulStrict(x)).mul(ys[j]);
                    x = x.sub(proj);
                }
            }
            return x.div(norm_1.norm(x, 'euclidean'));
        }));
    };
    for (var i = 0; i < xs.length; ++i) {
        _loop_1(i);
    }
    if (inputIsTensor2D) {
        return array_ops_1.stack(ys, 0);
    }
    else {
        return ys;
    }
}
function qr_(x, fullMatrices) {
    if (fullMatrices === void 0) { fullMatrices = false; }
    if (x.rank < 2) {
        throw new Error("qr() requires input tensor to have a rank >= 2, but got rank " + x.rank);
    }
    else if (x.rank === 2) {
        return qr2d(x, fullMatrices);
    }
    else {
        var outerDimsProd = x.shape.slice(0, x.shape.length - 2)
            .reduce(function (value, prev) { return value * prev; });
        var x2ds = array_ops_1.unstack(x.reshape([
            outerDimsProd, x.shape[x.shape.length - 2],
            x.shape[x.shape.length - 1]
        ]), 0);
        var q2ds_1 = [];
        var r2ds_1 = [];
        x2ds.forEach(function (x2d) {
            var _a = qr2d(x2d, fullMatrices), q2d = _a[0], r2d = _a[1];
            q2ds_1.push(q2d);
            r2ds_1.push(r2d);
        });
        var q = array_ops_1.stack(q2ds_1, 0).reshape(x.shape);
        var r = array_ops_1.stack(r2ds_1, 0).reshape(x.shape);
        return [q, r];
    }
}
function qr2d(x, fullMatrices) {
    if (fullMatrices === void 0) { fullMatrices = false; }
    return environment_1.ENV.engine.tidy(function () {
        if (x.shape.length !== 2) {
            throw new Error("qr2d() requires a 2D Tensor, but got a " + x.shape.length + "D Tensor.");
        }
        var m = x.shape[0];
        var n = x.shape[1];
        var q = array_ops_1.eye(m);
        var r = x.clone();
        var one2D = tensor_ops_1.tensor2d([[1]], [1, 1]);
        var w = one2D.clone();
        var iters = m >= n ? n : m;
        var _loop_2 = function (j) {
            var _a;
            var rTemp = r;
            var wTemp = w;
            var qTemp = q;
            _a = environment_1.ENV.engine.tidy(function () {
                var rjEnd1 = r.slice([j, j], [m - j, 1]);
                var normX = rjEnd1.norm();
                var rjj = r.slice([j, j], [1, 1]);
                var s = rjj.sign().neg();
                var u1 = rjj.sub(s.mul(normX));
                var wPre = rjEnd1.div(u1);
                if (wPre.shape[0] === 1) {
                    w = one2D.clone();
                }
                else {
                    w = one2D.concat(wPre.slice([1, 0], [wPre.shape[0] - 1, wPre.shape[1]]), 0);
                }
                var tau = s.matMul(u1).div(normX).neg();
                var rjEndAll = r.slice([j, 0], [m - j, n]);
                var tauTimesW = tau.mul(w);
                if (j === 0) {
                    r = rjEndAll.sub(tauTimesW.matMul(w.transpose().matMul(rjEndAll)));
                }
                else {
                    r = r.slice([0, 0], [j, n])
                        .concat(rjEndAll.sub(tauTimesW.matMul(w.transpose().matMul(rjEndAll))), 0);
                }
                var qAllJEnd = q.slice([0, j], [m, q.shape[1] - j]);
                if (j === 0) {
                    q = qAllJEnd.sub(qAllJEnd.matMul(w).matMul(tauTimesW.transpose()));
                }
                else {
                    q = q.slice([0, 0], [m, j])
                        .concat(qAllJEnd.sub(qAllJEnd.matMul(w).matMul(tauTimesW.transpose())), 1);
                }
                return [w, r, q];
            }), w = _a[0], r = _a[1], q = _a[2];
            globals_1.dispose([rTemp, wTemp, qTemp]);
        };
        for (var j = 0; j < iters; ++j) {
            _loop_2(j);
        }
        if (!fullMatrices && m > n) {
            q = q.slice([0, 0], [m, n]);
            r = r.slice([0, 0], [n, n]);
        }
        return [q, r];
    });
}
exports.gramSchmidt = operation_1.op({ gramSchmidt_: gramSchmidt_ });
exports.qr = operation_1.op({ qr_: qr_ });
//# sourceMappingURL=linalg_ops.js.map