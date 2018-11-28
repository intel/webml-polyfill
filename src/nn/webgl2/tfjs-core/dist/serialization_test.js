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
var optimizer_1 = require("./optimizers/optimizer");
var serialization_1 = require("./serialization");
describe('registerClass', function () {
    var randomClassName = "OptimizerForTest" + Math.random();
    var OptimizerForTest = (function (_super) {
        __extends(OptimizerForTest, _super);
        function OptimizerForTest() {
            return _super.call(this) || this;
        }
        OptimizerForTest.prototype.applyGradients = function (variableGradients) { };
        OptimizerForTest.prototype.getConfig = function () {
            return {};
        };
        OptimizerForTest.className = randomClassName;
        return OptimizerForTest;
    }(optimizer_1.Optimizer));
    it('registerClass succeeds', function () {
        serialization_1.registerClass(OptimizerForTest);
        expect(serialization_1.SerializationMap.getMap().classNameMap[randomClassName] != null)
            .toEqual(true);
    });
    var OptimizerWithoutClassName = (function (_super) {
        __extends(OptimizerWithoutClassName, _super);
        function OptimizerWithoutClassName() {
            return _super.call(this) || this;
        }
        OptimizerWithoutClassName.prototype.applyGradients = function (variableGradients) { };
        OptimizerWithoutClassName.prototype.getConfig = function () {
            return {};
        };
        return OptimizerWithoutClassName;
    }(optimizer_1.Optimizer));
    it('registerClass fails on missing className', function () {
        expect(function () { return serialization_1.registerClass(OptimizerWithoutClassName); })
            .toThrowError(/does not have the static className property/);
    });
    var OptimizerWithEmptyClassName = (function (_super) {
        __extends(OptimizerWithEmptyClassName, _super);
        function OptimizerWithEmptyClassName() {
            return _super.call(this) || this;
        }
        OptimizerWithEmptyClassName.prototype.applyGradients = function (variableGradients) { };
        OptimizerWithEmptyClassName.prototype.getConfig = function () {
            return {};
        };
        OptimizerWithEmptyClassName.className = '';
        return OptimizerWithEmptyClassName;
    }(optimizer_1.Optimizer));
    it('registerClass fails on missing className', function () {
        expect(function () { return serialization_1.registerClass(OptimizerWithEmptyClassName); })
            .toThrowError(/has an empty-string as its className/);
    });
    var OptimizerWithNonStringClassName = (function (_super) {
        __extends(OptimizerWithNonStringClassName, _super);
        function OptimizerWithNonStringClassName() {
            return _super.call(this) || this;
        }
        OptimizerWithNonStringClassName.prototype.applyGradients = function (variableGradients) { };
        OptimizerWithNonStringClassName.prototype.getConfig = function () {
            return {};
        };
        OptimizerWithNonStringClassName.className = 42;
        return OptimizerWithNonStringClassName;
    }(optimizer_1.Optimizer));
    it('registerClass fails on missing className', function () {
        expect(function () { return serialization_1.registerClass(OptimizerWithNonStringClassName); })
            .toThrowError(/is required to be a string, but got type number/);
    });
});
//# sourceMappingURL=serialization_test.js.map