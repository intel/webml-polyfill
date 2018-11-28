"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var contexts = {};
var WEBGL_ATTRIBUTES = {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    depth: false,
    stencil: false,
    failIfMajorPerformanceCaveat: true
};
function getWebGLContext(webGLVersion) {
    if (!(webGLVersion in contexts)) {
        var canvas = document.createElement('canvas');
        canvas.addEventListener('webglcontextlost', function (ev) {
            ev.preventDefault();
            delete contexts[webGLVersion];
        }, false);
        contexts[webGLVersion] = getWebGLRenderingContext(webGLVersion);
    }
    var gl = contexts[webGLVersion];
    if (gl.isContextLost()) {
        delete contexts[webGLVersion];
        return getWebGLContext(webGLVersion);
    }
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    gl.disable(gl.SAMPLE_COVERAGE);
    gl.enable(gl.SCISSOR_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    return contexts[webGLVersion];
}
exports.getWebGLContext = getWebGLContext;
function getWebGLRenderingContext(webGLVersion) {
    if (webGLVersion !== 1 && webGLVersion !== 2) {
        throw new Error('Cannot get WebGL rendering context, WebGL is disabled.');
    }
    var canvas = document.createElement('canvas');
    if (webGLVersion === 1) {
        return (canvas.getContext('webgl', WEBGL_ATTRIBUTES) ||
            canvas.getContext('experimental-webgl', WEBGL_ATTRIBUTES));
    }
    return canvas.getContext('webgl2', WEBGL_ATTRIBUTES);
}
//# sourceMappingURL=canvas_util.js.map