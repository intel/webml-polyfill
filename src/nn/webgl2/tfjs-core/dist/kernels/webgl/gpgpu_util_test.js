"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../../index");
var jasmine_util_1 = require("../../jasmine_util");
var test_util_1 = require("../../test_util");
var gpgpu_context_1 = require("./gpgpu_context");
var gpgpu_util = require("./gpgpu_util");
var DOWNLOAD_FLOAT_ENVS = {
    'WEBGL_DOWNLOAD_FLOAT_ENABLED': true
};
jasmine_util_1.describeWithFlags('gpgpu_util createWebGLContext', test_util_1.WEBGL_ENVS, function () {
    var gpgpu;
    beforeEach(function () {
        gpgpu = new gpgpu_context_1.GPGPUContext();
    });
    afterEach(function () {
        gpgpu.dispose();
    });
    it('disables DEPTH_TEST and STENCIL_TEST', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.DEPTH_TEST)).toEqual(false);
        expect(gpgpu.gl.getParameter(gpgpu.gl.STENCIL_TEST)).toEqual(false);
    });
    it('disables BLEND', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.BLEND)).toEqual(false);
    });
    it('disables DITHER, POLYGON_OFFSET_FILL', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.DITHER)).toEqual(false);
        expect(gpgpu.gl.getParameter(gpgpu.gl.POLYGON_OFFSET_FILL)).toEqual(false);
    });
    it('enables CULL_FACE with BACK', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE)).toEqual(true);
        expect(gpgpu.gl.getParameter(gpgpu.gl.CULL_FACE_MODE))
            .toEqual(gpgpu.gl.BACK);
    });
    it('enables SCISSOR_TEST', function () {
        expect(gpgpu.gl.getParameter(gpgpu.gl.SCISSOR_TEST)).toEqual(true);
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util createFloat32MatrixTexture', test_util_1.WEBGL_ENVS, function () {
    it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
    it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createFloat32MatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util createPackedMatrixTexture', test_util_1.WEBGL_ENVS, function () {
    it('sets the TEXTURE_WRAP S+T parameters to CLAMP_TO_EDGE', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_S))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_WRAP_T))
            .toEqual(gpgpu.gl.CLAMP_TO_EDGE);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
    it('sets the TEXTURE_[MIN|MAG]_FILTER parameters to NEAREST', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 32, 32, textureConfig);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MIN_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        expect(gpgpu.gl.getTexParameter(gpgpu.gl.TEXTURE_2D, gpgpu.gl.TEXTURE_MAG_FILTER))
            .toEqual(gpgpu.gl.NEAREST);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, null);
        gpgpu.deleteMatrixTexture(tex);
        gpgpu.dispose();
    });
});
jasmine_util_1.describeWithFlags('gpgpu_util downloadMatrixFromPackedOutputTexture', DOWNLOAD_FLOAT_ENVS, function () {
    it('should work when texture shape != logical shape', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, 4, 6, textureConfig);
        var mat = tf.tensor2d([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 12]);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        gpgpu.gl.texSubImage2D(gpgpu.gl.TEXTURE_2D, 0, 0, 0, 3, 2, gpgpu.gl.RGBA, gpgpu.gl.FLOAT, new Float32Array([
            0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0,
            6, 7, 0, 0, 8, 9, 0, 0, 10, 11, 0, 0
        ]));
        var result = gpgpu.downloadMatrixFromPackedTexture(tex, 1, 1, 12, 4, 6);
        test_util_1.expectArraysClose(result, mat.dataSync());
    });
    it('should work when different batches occupy the same physical row', function () {
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var textureConfig = gpgpu_util.getTextureConfig(gpgpu.gl);
        var physicalRows = 10;
        var physicalCols = 16;
        var tex = gpgpu_util.createPackedMatrixTexture(gpgpu.gl, physicalRows, physicalCols, textureConfig);
        var mat = tf.tensor3d([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
            73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
        ], [2, 20, 3]);
        gpgpu.gl.bindTexture(gpgpu.gl.TEXTURE_2D, tex);
        gpgpu.gl.texSubImage2D(gpgpu.gl.TEXTURE_2D, 0, 0, 0, 8, 5, gpgpu.gl.RGBA, gpgpu.gl.FLOAT, new Float32Array([
            1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11,
            9, 0, 12, 0, 13, 14, 16, 17, 15, 0, 18, 0,
            19, 20, 22, 23, 21, 0, 24, 0, 25, 26, 28, 29,
            27, 0, 30, 0, 31, 32, 34, 35, 33, 0, 36, 0,
            37, 38, 40, 41, 39, 0, 42, 0, 43, 44, 46, 47,
            45, 0, 48, 0, 49, 50, 52, 53, 51, 0, 54, 0,
            55, 56, 58, 59, 57, 0, 60, 0, 61, 62, 64, 65,
            63, 0, 66, 0, 67, 68, 70, 71, 69, 0, 72, 0,
            73, 74, 76, 77, 75, 0, 78, 0, 79, 80, 82, 83,
            81, 0, 84, 0, 85, 86, 88, 89, 87, 0, 90, 0,
            91, 92, 94, 95, 93, 0, 96, 0, 97, 98, 100, 101,
            99, 0, 102, 0, 103, 104, 106, 107, 105, 0, 108, 0,
            109, 110, 112, 113, 111, 0, 114, 0, 115, 116, 118, 119,
            117, 0, 120, 0
        ]));
        var result = gpgpu.downloadMatrixFromPackedTexture(tex, 2, 20, 3, physicalRows, physicalCols);
        test_util_1.expectArraysClose(result, mat.dataSync());
    });
});
//# sourceMappingURL=gpgpu_util_test.js.map