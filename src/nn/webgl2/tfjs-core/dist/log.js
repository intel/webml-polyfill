"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("./environment");
function warn() {
    var msg = [];
    for (var _i = 0; _i < arguments.length; _i++) {
        msg[_i] = arguments[_i];
    }
    if (!environment_1.ENV.get('IS_TEST')) {
        console.warn.apply(console, msg);
    }
}
exports.warn = warn;
function log() {
    var msg = [];
    for (var _i = 0; _i < arguments.length; _i++) {
        msg[_i] = arguments[_i];
    }
    if (!environment_1.ENV.get('IS_TEST')) {
        console.log.apply(console, msg);
    }
}
exports.log = log;
//# sourceMappingURL=log.js.map