"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var util = require("../../util");
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
function createVertexShader(gl) {
    var vertexShaderSource = "#version 300 es\n    precision highp float;\n\n    in vec3 clipSpacePos;\n    in vec2 uv;\n    out vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }";
    return webgl_util.createVertexShader(gl, vertexShaderSource);
}
exports.createVertexShader = createVertexShader;
function createVertexBuffer(gl) {
    var vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
    return webgl_util.createStaticVertexBuffer(gl, vertexArray);
}
exports.createVertexBuffer = createVertexBuffer;
function createIndexBuffer(gl) {
    var triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
    return webgl_util.createStaticIndexBuffer(gl, triangleVertexIndices);
}
exports.createIndexBuffer = createIndexBuffer;
function getTextureConfig(gl, textureHalfFloatExtension) {
    var glany = gl;
    var internalFormatFloat;
    var internalFormatHalfFloat;
    var internalFormatPackedFloat;
    var textureFormatFloat;
    var downloadTextureFormat;
    var downloadUnpackNumChannels;
    var defaultNumChannels;
    var textureTypeHalfFloat;
    if (environment_1.ENV.get('WEBGL_VERSION') === 2) {
        internalFormatFloat = glany.R32F;
        internalFormatHalfFloat = glany.R16F;
        internalFormatPackedFloat = glany.RGBA32F;
        textureFormatFloat = glany.RED;
        downloadUnpackNumChannels = 4;
        defaultNumChannels = 1;
        textureTypeHalfFloat = glany.HALF_FLOAT;
    }
    else {
        internalFormatFloat = gl.RGBA;
        internalFormatHalfFloat = gl.RGBA;
        internalFormatPackedFloat = glany.RGBA;
        textureFormatFloat = gl.RGBA;
        downloadUnpackNumChannels = 4;
        defaultNumChannels = 4;
        textureTypeHalfFloat = textureHalfFloatExtension != null ?
            textureHalfFloatExtension.HALF_FLOAT_OES :
            null;
    }
    downloadTextureFormat = gl.RGBA;
    return {
        internalFormatFloat: internalFormatFloat,
        internalFormatHalfFloat: internalFormatHalfFloat,
        internalFormatPackedFloat: internalFormatPackedFloat,
        textureFormatFloat: textureFormatFloat,
        downloadTextureFormat: downloadTextureFormat,
        downloadUnpackNumChannels: downloadUnpackNumChannels,
        defaultNumChannels: defaultNumChannels,
        textureTypeHalfFloat: textureTypeHalfFloat
    };
}
exports.getTextureConfig = getTextureConfig;
function createAndConfigureTexture(gl, width, height, internalFormat, textureFormat, textureType) {
    webgl_util.validateTextureSize(width, height);
    var texture = webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, textureFormat, textureType, null); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
function createFloat32MatrixTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    return createAndConfigureTexture(gl, width, height, textureConfig.internalFormatFloat, textureConfig.textureFormatFloat, gl.FLOAT);
}
exports.createFloat32MatrixTexture = createFloat32MatrixTexture;
function createFloat16MatrixTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    return createAndConfigureTexture(gl, width, height, textureConfig.internalFormatFloat, textureConfig.textureFormatFloat, textureConfig.textureTypeHalfFloat);
}
exports.createFloat16MatrixTexture = createFloat16MatrixTexture;
function createUnsignedBytesMatrixTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    return createAndConfigureTexture(gl, width, height, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE);
}
exports.createUnsignedBytesMatrixTexture = createUnsignedBytesMatrixTexture;
function createPackedMatrixTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    return createAndConfigureTexture(gl, width, height, textureConfig.internalFormatPackedFloat, gl.RGBA, gl.FLOAT);
}
exports.createPackedMatrixTexture = createPackedMatrixTexture;
function createFloat16PackedMatrixTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    return createAndConfigureTexture(gl, width, height, textureConfig.internalFormatHalfFloat, gl.RGBA, textureConfig.textureTypeHalfFloat);
}
exports.createFloat16PackedMatrixTexture = createFloat16PackedMatrixTexture;
function bindVertexProgramAttributeStreams(gl, program, vertexBuffer) {
    var posOffset = 0;
    var uvOffset = 3 * 4;
    var stride = (3 * 4) + (2 * 4);
    webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer); });
    var success = webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
    return success &&
        webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
}
exports.bindVertexProgramAttributeStreams = bindVertexProgramAttributeStreams;
function uploadPixelDataToTexture(gl, texture, pixels) {
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, pixels); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.uploadPixelDataToTexture = uploadPixelDataToTexture;
function uploadDataToTexture(gl, texture, width, height, data, textureFormat) {
    webgl_util.validateTextureSize(width, height);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, textureFormat, gl.FLOAT, data); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
function uploadMatrixToTexture(gl, texture, rows, columns, matrix, numChannels, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var unpackedArray;
    if (textureConfig.defaultNumChannels === 1) {
        unpackedArray = matrix;
    }
    else {
        unpackedArray =
            new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, numChannels));
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, numChannels);
    }
    uploadDataToTexture(gl, texture, w, h, unpackedArray, textureConfig.textureFormatFloat);
}
exports.uploadMatrixToTexture = uploadMatrixToTexture;
function uploadMatrixToPackedTexture(gl, texture, batch, rows, columns, matrix, textureConfig) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    tex_util.encodeMatrixToPackedRGBA(matrix, batch, rows, columns, packedRGBA);
    uploadDataToTexture(gl, texture, w, h, packedRGBA, gl.RGBA);
}
exports.uploadMatrixToPackedTexture = uploadMatrixToPackedTexture;
function maybeCreateBufferFromOutputTexture(gl, texture, rows, columns, textureConfig) {
    var bufferOrTexture = texture;
    if (environment_1.ENV.get('WEBGL_VERSION') === 2) {
        var gl2_1 = gl;
        var buffer_1 = gl2_1.createBuffer();
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl2_1.PIXEL_PACK_BUFFER, buffer_1); });
        var bytesPerFloat = 4;
        var bufferSizeBytes_1 = bytesPerFloat *
            tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, textureConfig.downloadUnpackNumChannels);
        webgl_util.callAndCheck(gl, function () { return gl.bufferData(gl2_1.PIXEL_PACK_BUFFER, bufferSizeBytes_1, gl.STATIC_DRAW); });
        webgl_util.callAndCheck(gl, function () { return gl2_1.readPixels(0, 0, columns, rows, gl.RGBA, gl.FLOAT, 0); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl2_1.PIXEL_PACK_BUFFER, null); });
        bufferOrTexture = buffer_1;
    }
    return bufferOrTexture;
}
exports.maybeCreateBufferFromOutputTexture = maybeCreateBufferFromOutputTexture;
function downloadFloat32MatrixFromBuffer(gl, buffer, rows, columns, textureConfig) {
    var gl2 = gl;
    var downloadTarget = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, textureConfig.downloadUnpackNumChannels));
    gl2.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl2.getBufferSubData(gl.ARRAY_BUFFER, 0, downloadTarget);
    gl2.bindBuffer(gl.ARRAY_BUFFER, null);
    var matrix = new Float32Array(rows * columns);
    tex_util.decodeMatrixFromUnpackedArray(downloadTarget, matrix, textureConfig.downloadUnpackNumChannels);
    return matrix;
}
exports.downloadFloat32MatrixFromBuffer = downloadFloat32MatrixFromBuffer;
function downloadFloat32MatrixFromOutputTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var downloadTarget = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, textureConfig.downloadUnpackNumChannels));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, textureConfig.downloadTextureFormat, gl.FLOAT, downloadTarget); });
    var matrix = new Float32Array(rows * columns);
    tex_util.decodeMatrixFromUnpackedArray(downloadTarget, matrix, textureConfig.downloadUnpackNumChannels);
    return matrix;
}
exports.downloadFloat32MatrixFromOutputTexture = downloadFloat32MatrixFromOutputTexture;
function downloadByteEncodedFloatMatrixFromOutputTexture(gl, rows, columns, textureConfig) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var numChannels = 4;
    var downloadTarget = new Uint8Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, numChannels));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, textureConfig.downloadTextureFormat, gl.UNSIGNED_BYTE, downloadTarget); });
    return new Float32Array(downloadTarget.buffer);
}
exports.downloadByteEncodedFloatMatrixFromOutputTexture = downloadByteEncodedFloatMatrixFromOutputTexture;
function downloadMatrixFromPackedOutputTexture(gl, batch, rows, cols, physicalRows, physicalCols, textureConfig) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(physicalRows, physicalCols), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(physicalRows, physicalCols));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, packedRGBA); });
    var matrix = new Float32Array(util.sizeFromShape([batch, rows, cols]));
    return tex_util.decodeMatrixFromPackedRGBA(packedRGBA, batch, rows, cols, matrix);
}
exports.downloadMatrixFromPackedOutputTexture = downloadMatrixFromPackedOutputTexture;
//# sourceMappingURL=gpgpu_util.js.map