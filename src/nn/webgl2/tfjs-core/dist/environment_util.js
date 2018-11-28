"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var canvas_util_1 = require("./canvas_util");
var Type;
(function (Type) {
    Type[Type["NUMBER"] = 0] = "NUMBER";
    Type[Type["BOOLEAN"] = 1] = "BOOLEAN";
    Type[Type["STRING"] = 2] = "STRING";
})(Type = exports.Type || (exports.Type = {}));
exports.URL_PROPERTIES = [
    { name: 'DEBUG', type: Type.BOOLEAN },
    { name: 'IS_BROWSER', type: Type.BOOLEAN },
    { name: 'WEBGL_LAZILY_UNPACK', type: Type.BOOLEAN },
    { name: 'WEBGL_CPU_FORWARD', type: Type.BOOLEAN },
    { name: 'WEBGL_PACK_BATCHNORMALIZATION', type: Type.BOOLEAN },
    { name: 'WEBGL_CONV_IM2COL', type: Type.BOOLEAN },
    { name: 'WEBGL_MAX_TEXTURE_SIZE', type: Type.NUMBER },
    { name: 'WEBGL_PAGING_ENABLED', type: Type.BOOLEAN },
    { name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION', type: Type.NUMBER },
    { name: 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', type: Type.BOOLEAN },
    { name: 'WEBGL_VERSION', type: Type.NUMBER },
    { name: 'WEBGL_RENDER_FLOAT32_ENABLED', type: Type.BOOLEAN },
    { name: 'WEBGL_DOWNLOAD_FLOAT_ENABLED', type: Type.BOOLEAN },
    { name: 'WEBGL_FENCE_API_ENABLED', type: Type.BOOLEAN },
    { name: 'WEBGL_SIZE_UPLOAD_UNIFORM', type: Type.NUMBER },
    { name: 'BACKEND', type: Type.STRING },
    { name: 'EPSILON', type: Type.NUMBER },
    { name: 'PROD', type: Type.BOOLEAN },
    { name: 'TENSORLIKE_CHECK_SHAPE_CONSISTENCY', type: Type.BOOLEAN },
];
function isWebGLVersionEnabled(webGLVersion) {
    try {
        var gl = canvas_util_1.getWebGLContext(webGLVersion);
        if (gl != null) {
            return true;
        }
    }
    catch (e) {
        return false;
    }
    return false;
}
exports.isWebGLVersionEnabled = isWebGLVersionEnabled;
var MAX_TEXTURE_SIZE;
function getWebGLMaxTextureSize(webGLVersion) {
    if (MAX_TEXTURE_SIZE == null) {
        var gl = canvas_util_1.getWebGLContext(webGLVersion);
        MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    }
    return MAX_TEXTURE_SIZE;
}
exports.getWebGLMaxTextureSize = getWebGLMaxTextureSize;
function getWebGLDisjointQueryTimerVersion(webGLVersion) {
    if (webGLVersion === 0) {
        return 0;
    }
    var queryTimerVersion;
    var gl = canvas_util_1.getWebGLContext(webGLVersion);
    if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
        webGLVersion === 2) {
        queryTimerVersion = 2;
    }
    else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
        queryTimerVersion = 1;
    }
    else {
        queryTimerVersion = 0;
    }
    return queryTimerVersion;
}
exports.getWebGLDisjointQueryTimerVersion = getWebGLDisjointQueryTimerVersion;
function isRenderToFloatTextureEnabled(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    var gl = canvas_util_1.getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
    }
    else {
        if (!hasExtension(gl, 'EXT_color_buffer_float')) {
            return false;
        }
    }
    var isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl, webGLVersion);
    return isFrameBufferComplete;
}
exports.isRenderToFloatTextureEnabled = isRenderToFloatTextureEnabled;
function isDownloadFloatTextureEnabled(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    var gl = canvas_util_1.getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
        if (!hasExtension(gl, 'WEBGL_color_buffer_float')) {
            return false;
        }
    }
    else {
        if (!hasExtension(gl, 'EXT_color_buffer_float')) {
            return false;
        }
    }
    var isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl, webGLVersion);
    return isFrameBufferComplete;
}
exports.isDownloadFloatTextureEnabled = isDownloadFloatTextureEnabled;
function isWebGLFenceEnabled(webGLVersion) {
    if (webGLVersion !== 2) {
        return false;
    }
    var gl = canvas_util_1.getWebGLContext(webGLVersion);
    var isEnabled = gl.fenceSync != null;
    return isEnabled;
}
exports.isWebGLFenceEnabled = isWebGLFenceEnabled;
function isChrome() {
    return typeof navigator !== 'undefined' && navigator != null &&
        navigator.userAgent != null && /Chrome/.test(navigator.userAgent) &&
        /Google Inc/.test(navigator.vendor);
}
exports.isChrome = isChrome;
var TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
function getFeaturesFromURL() {
    var features = {};
    if (typeof window === 'undefined' || typeof window.location === 'undefined' ||
        typeof window.location.search === 'undefined') {
        return features;
    }
    var urlParams = getQueryParams(window.location.search);
    if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
        var urlFlags_1 = {};
        var keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
        keyValues.forEach(function (keyValue) {
            var _a = keyValue.split(':'), key = _a[0], value = _a[1];
            urlFlags_1[key] = value;
        });
        exports.URL_PROPERTIES.forEach(function (urlProperty) {
            if (urlProperty.name in urlFlags_1) {
                console.log("Setting feature override from URL " + urlProperty.name + ": " +
                    ("" + urlFlags_1[urlProperty.name]));
                if (urlProperty.type === Type.NUMBER) {
                    features[urlProperty.name] = +urlFlags_1[urlProperty.name];
                }
                else if (urlProperty.type === Type.BOOLEAN) {
                    features[urlProperty.name] = urlFlags_1[urlProperty.name] === 'true';
                }
                else if (urlProperty.type === Type.STRING) {
                    features[urlProperty.name] = urlFlags_1[urlProperty.name];
                }
                else {
                    console.warn("Unknown URL param: " + urlProperty.name + ".");
                }
            }
        });
    }
    return features;
}
exports.getFeaturesFromURL = getFeaturesFromURL;
function hasExtension(gl, extensionName) {
    var ext = gl.getExtension(extensionName);
    return ext != null;
}
function createFloatTextureAndBindToFramebuffer(gl, webGLVersion) {
    var frameBuffer = gl.createFramebuffer();
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var internalFormat = webGLVersion === 2 ? gl.RGBA32F : gl.RGBA;
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    var isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isFrameBufferComplete;
}
function getQueryParams(queryString) {
    var params = {};
    queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, function (s) {
        var t = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            t[_i - 1] = arguments[_i];
        }
        decodeParam(params, t[0], t[1]);
        return t.join('=');
    });
    return params;
}
exports.getQueryParams = getQueryParams;
function decodeParam(params, name, value) {
    params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
//# sourceMappingURL=environment_util.js.map