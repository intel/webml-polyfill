"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../../index");
var jasmine_util_1 = require("../../jasmine_util");
var test_util_1 = require("../../test_util");
jasmine_util_1.describeWithFlags('custom-op webgl', test_util_1.WEBGL_ENVS, function () {
    var SquareAndAddKernel = (function () {
        function SquareAndAddKernel(inputShape) {
            this.variableNames = ['X'];
            this.outputShape = inputShape.slice();
            this.userCode = "\n          void main() {\n            float x = getXAtOutCoords();\n            float value = x * x + x;\n            setOutput(value);\n          }\n        ";
        }
        return SquareAndAddKernel;
    }());
    var SquareAndAddBackpropKernel = (function () {
        function SquareAndAddBackpropKernel(inputShape) {
            this.variableNames = ['X'];
            this.outputShape = inputShape.slice();
            this.userCode = "\n          void main() {\n            float x = getXAtOutCoords();\n            float value = 2.0 * x + 1.0;\n            setOutput(value);\n          }\n        ";
        }
        return SquareAndAddBackpropKernel;
    }());
    function squareAndAdd(x) {
        var fn = tf.customGrad(function (x) {
            var webglBackend = tf.ENV.backend;
            var program = new SquareAndAddKernel(x.shape);
            var backpropProgram = new SquareAndAddBackpropKernel(x.shape);
            var value = webglBackend.compileAndRun(program, [x]);
            var gradFunc = function (dy) {
                return webglBackend.compileAndRun(backpropProgram, [x]).mul(dy);
            };
            return { value: value, gradFunc: gradFunc };
        });
        return fn(x);
    }
    it('lets users use custom operations', function () {
        var inputArr = [1, 2, 3, 4];
        var input = tf.tensor(inputArr);
        var output = squareAndAdd(input);
        test_util_1.expectArraysClose(output, inputArr.map(function (x) { return x * x + x; }));
    });
    it('lets users define gradients for operations', function () {
        var inputArr = [1, 2, 3, 4];
        var input = tf.tensor(inputArr);
        var grads = tf.valueAndGrad(function (x) { return squareAndAdd(x); });
        var _a = grads(input), value = _a.value, grad = _a.grad;
        test_util_1.expectArraysClose(value, inputArr.map(function (x) { return x * x + x; }));
        test_util_1.expectArraysClose(grad, inputArr.map(function (x) { return 2 * x + 1; }));
    });
    it('multiplies by dy parameter when it is passed', function () {
        var inputArr = [1, 2, 3, 4];
        var input = tf.tensor(inputArr);
        var grads = tf.valueAndGrad(function (x) { return squareAndAdd(x); });
        var _a = grads(input, tf.zerosLike(input)), value = _a.value, grad = _a.grad;
        test_util_1.expectArraysClose(value, inputArr.map(function (x) { return x * x + x; }));
        test_util_1.expectArraysClose(grad, inputArr.map(function () { return 0.0; }));
    });
});
//# sourceMappingURL=webgl_custom_op_test.js.map