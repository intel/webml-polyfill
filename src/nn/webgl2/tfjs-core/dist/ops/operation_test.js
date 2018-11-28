"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var operation_1 = require("./operation");
jasmine_util_1.describeWithFlags('operation', test_util_1.ALL_ENVS, function () {
    it('executes and preserves function name', function () {
        var f = function () { return 2; };
        var opfn = operation_1.op({ 'opName': f });
        expect(opfn.name).toBe('opName');
        expect(opfn()).toBe(2);
    });
    it('executes, preserves function name, strips underscore', function () {
        var f = function () { return 2; };
        var opfn = operation_1.op({ 'opName_': f });
        expect(opfn.name).toBe('opName');
        expect(opfn()).toBe(2);
    });
    it('throws when passing an object with multiple keys', function () {
        var f = function () { return 2; };
        expect(function () { return operation_1.op({ 'opName_': f, 'opName2_': f }); })
            .toThrowError(/Please provide an object with a single key/);
    });
});
//# sourceMappingURL=operation_test.js.map