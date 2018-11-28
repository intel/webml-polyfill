"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var globals_1 = require("../globals");
var serialization_1 = require("../serialization");
var Optimizer = (function (_super) {
    __extends(Optimizer, _super);
    function Optimizer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    Optimizer.prototype.minimize = function (f, returnCost, varList) {
        if (returnCost === void 0) { returnCost = false; }
        var _a = this.computeGradients(f, varList), value = _a.value, grads = _a.grads;
        this.applyGradients(grads);
        var varNames = Object.keys(grads);
        varNames.forEach(function (varName) { return grads[varName].dispose(); });
        if (returnCost) {
            return value;
        }
        else {
            value.dispose();
            return null;
        }
    };
    Optimizer.prototype.computeGradients = function (f, varList) {
        return globals_1.variableGrads(f, varList);
    };
    return Optimizer;
}(serialization_1.Serializable));
exports.Optimizer = Optimizer;
//# sourceMappingURL=optimizer.js.map