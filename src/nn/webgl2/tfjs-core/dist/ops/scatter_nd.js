"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
var tensor_util_env_1 = require("../tensor_util_env");
var operation_1 = require("./operation");
var scatter_nd_util = require("./scatter_nd_util");
function scatterND_(indices, updates, shape) {
    var $indices = tensor_util_env_1.convertToTensor(indices, 'indices', 'scatterND', 'int32');
    var $updates = tensor_util_env_1.convertToTensor(updates, 'updates', 'scatterND');
    scatter_nd_util.validateInput($updates, $indices, shape);
    return environment_1.ENV.engine.runKernel(function (backend) { return backend.scatterND($indices, $updates, shape); }, { $indices: $indices, $updates: $updates });
}
exports.scatterND = operation_1.op({ scatterND_: scatterND_ });
//# sourceMappingURL=scatter_nd.js.map