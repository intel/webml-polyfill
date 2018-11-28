"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("./environment");
var gradients_1 = require("./gradients");
exports.customGrad = gradients_1.customGrad;
exports.grad = gradients_1.grad;
exports.grads = gradients_1.grads;
exports.valueAndGrad = gradients_1.valueAndGrad;
exports.valueAndGrads = gradients_1.valueAndGrads;
exports.variableGrads = gradients_1.variableGrads;
exports.tidy = environment_1.Environment.tidy;
exports.keep = environment_1.Environment.keep;
exports.dispose = environment_1.Environment.dispose;
exports.time = environment_1.Environment.time;
exports.profile = environment_1.Environment.profile;
//# sourceMappingURL=globals.js.map