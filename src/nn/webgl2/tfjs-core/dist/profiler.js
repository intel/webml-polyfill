"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("./util");
var Profiler = (function () {
    function Profiler(backendTimer, logger) {
        this.backendTimer = backendTimer;
        this.logger = logger;
        if (logger == null) {
            this.logger = new Logger();
        }
    }
    Profiler.prototype.profileKernel = function (name, f) {
        var _this = this;
        var result;
        var holdResultWrapperFn = function () {
            result = f();
        };
        var timer = this.backendTimer.time(holdResultWrapperFn);
        var results = Array.isArray(result) ? result : [result];
        results.forEach(function (r) {
            var vals = r.dataSync();
            util.checkComputationForNaN(vals, r.dtype, name);
            timer.then(function (timing) {
                var extraInfo = '';
                if (timing.getExtraProfileInfo != null) {
                    extraInfo = timing.getExtraProfileInfo();
                }
                _this.logger.logKernelProfile(name, r, vals, timing.kernelMs, extraInfo);
            });
        });
        return result;
    };
    return Profiler;
}());
exports.Profiler = Profiler;
var Logger = (function () {
    function Logger() {
    }
    Logger.prototype.logKernelProfile = function (name, result, vals, timeMs, extraInfo) {
        var time = util.rightPad(timeMs + "ms", 9);
        var paddedName = util.rightPad(name, 25);
        var rank = result.rank;
        var size = result.size;
        var shape = util.rightPad(result.shape.toString(), 14);
        console.log("%c" + paddedName + "\t%c" + time + "\t%c" + rank + "D " + shape + "\t%c" + size + "\t%c" + extraInfo, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange', 'color: green');
    };
    return Logger;
}());
exports.Logger = Logger;
//# sourceMappingURL=profiler.js.map