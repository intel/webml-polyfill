"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util_1 = require("./util");
var Serializable = (function () {
    function Serializable() {
    }
    Serializable.prototype.getClassName = function () {
        return this.constructor
            .className;
    };
    Serializable.fromConfig = function (cls, config) {
        return new cls(config);
    };
    return Serializable;
}());
exports.Serializable = Serializable;
var SerializationMap = (function () {
    function SerializationMap() {
        this.classNameMap = {};
    }
    SerializationMap.getMap = function () {
        if (SerializationMap.instance == null) {
            SerializationMap.instance = new SerializationMap();
        }
        return SerializationMap.instance;
    };
    SerializationMap.register = function (cls) {
        SerializationMap.getMap().classNameMap[cls.className] =
            [cls, cls.fromConfig];
    };
    return SerializationMap;
}());
exports.SerializationMap = SerializationMap;
function registerClass(cls) {
    util_1.assert(cls.className != null, "Class being registered does not have the static className property " +
        "defined.");
    util_1.assert(typeof cls.className === 'string', "className is required to be a string, but got type " +
        typeof cls.className);
    util_1.assert(cls.className.length > 0, "Class being registered has an empty-string as its className, which " +
        "is disallowed.");
    SerializationMap.register(cls);
}
exports.registerClass = registerClass;
//# sourceMappingURL=serialization.js.map