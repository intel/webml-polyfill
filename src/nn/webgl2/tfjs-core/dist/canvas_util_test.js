"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var canvas_util_1 = require("./canvas_util");
var environment_1 = require("./environment");
var jasmine_util_1 = require("./jasmine_util");
var test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('canvas_util', test_util_1.BROWSER_ENVS, function () {
    it('Returns a valid canvas', function () {
        var canvas = canvas_util_1.getWebGLContext(environment_1.ENV.get('WEBGL_VERSION')).canvas;
        expect(canvas instanceof HTMLCanvasElement).toBe(true);
    });
    it('Returns a valid gl context', function () {
        var gl = canvas_util_1.getWebGLContext(environment_1.ENV.get('WEBGL_VERSION'));
        expect(gl.isContextLost()).toBe(false);
    });
});
jasmine_util_1.describeWithFlags('canvas_util webgl2', { WEBGL_VERSION: 2 }, function () {
    it('is ok when the user requests webgl 1 canvas', function () {
        var canvas = canvas_util_1.getWebGLContext(1).canvas;
        expect(canvas instanceof HTMLCanvasElement).toBe(true);
    });
});
//# sourceMappingURL=canvas_util_test.js.map