"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../environment");
function op(f) {
    var keys = Object.keys(f);
    if (keys.length !== 1) {
        throw new Error("Please provide an object with a single key " +
            "(operation name) mapping to a function. Got an object with " +
            (keys.length + " keys."));
    }
    var opName = keys[0];
    var fn = f[opName];
    if (opName.endsWith('_')) {
        opName = opName.substring(0, opName.length - 1);
    }
    var f2 = function () {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i] = arguments[_i];
        }
        environment_1.ENV.engine.startScope(opName);
        try {
            var result = fn.apply(void 0, args);
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            environment_1.ENV.engine.endScope(result);
            return result;
        }
        catch (ex) {
            environment_1.ENV.engine.endScope(null);
            throw ex;
        }
    };
    Object.defineProperty(f2, 'name', { value: opName, configurable: true });
    return f2;
}
exports.op = op;
//# sourceMappingURL=operation.js.map