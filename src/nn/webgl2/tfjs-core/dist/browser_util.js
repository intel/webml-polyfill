"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var delayCallback = typeof requestAnimationFrame !== 'undefined' ?
    requestAnimationFrame :
    setImmediate;
function nextFrame() {
    return new Promise(function (resolve) { return delayCallback(function () { return resolve(); }); });
}
exports.nextFrame = nextFrame;
//# sourceMappingURL=browser_util.js.map