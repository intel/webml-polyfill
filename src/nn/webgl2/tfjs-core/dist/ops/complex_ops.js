"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var util = require("../util");
var operation_1 = require("./operation");
function complex_(real, imag) {
    var $real = tensor_util_env_1.convertToTensor(real, 'real', 'complex');
    var $imag = tensor_util_env_1.convertToTensor(imag, 'imag', 'complex');
    util.assertShapesMatch($real.shape, $imag.shape, "real and imag shapes, " + $real.shape + " and " + $imag.shape + ", " +
        "must match in call to tf.complex().");
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.complex($real, $imag); }, { $real: $real, $imag: $imag });
}
function real_(input) {
    var $input = tensor_util_env_1.convertToTensor(input, 'input', 'real');
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.real($input); }, { $input: $input });
}
function imag_(input) {
    var $input = tensor_util_env_1.convertToTensor(input, 'input', 'imag');
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.imag($input); }, { $input: $input });
}
exports.complex = operation_1.op({ complex_: complex_ });
exports.real = operation_1.op({ real_: real_ });
exports.imag = operation_1.op({ imag_: imag_ });
//# sourceMappingURL=complex_ops.js.map