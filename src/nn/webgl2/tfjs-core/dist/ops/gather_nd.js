"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
function gatherND_(x, indices) {
    var $indices = tensor_util_env_1.convertToTensor(indices, 'indices', 'gatherND', 'int32');
    var $x = tensor_util_env_1.convertToTensor(x, 'x', 'gatherND');
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.gatherND($x, $indices); }, { $x: $x, $indices: $indices });
}
exports.gatherND = operation_1.op({ gatherND_: gatherND_ });
//# sourceMappingURL=gather_nd.js.map