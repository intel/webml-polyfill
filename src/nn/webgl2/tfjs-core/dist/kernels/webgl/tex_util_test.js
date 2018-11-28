"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../../jasmine_util");
var test_util_1 = require("../../test_util");
var tex_util = require("./tex_util");
describe('tex_util getUnpackedMatrixTextureShapeWidthHeight', function () {
    it('[1x1] => [1x1]', function () {
        expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(1, 1)).toEqual([
            1, 1
        ]);
    });
    it('[MxN] => [NxM]', function () {
        expect(tex_util.getUnpackedMatrixTextureShapeWidthHeight(123, 456))
            .toEqual([456, 123]);
    });
});
describe('tex_util getPackedMatrixTextureShapeWidthHeight', function () {
    it('[1x1] => [1x1]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 1);
        expect(shape).toEqual([1, 1]);
    });
    it('[1x2] => [1x1]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1, 2);
        expect(shape).toEqual([1, 1]);
    });
    it('[2x1] => [1x1]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 1);
        expect(shape).toEqual([1, 1]);
    });
    it('[2x2] => [1x1]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(2, 2);
        expect(shape).toEqual([1, 1]);
    });
    it('[3x3] => [2x2]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 3);
        expect(shape).toEqual([2, 2]);
    });
    it('[4x3] => [2x2]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 3);
        expect(shape).toEqual([2, 2]);
    });
    it('[3x4] => [2x2]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(3, 4);
        expect(shape).toEqual([2, 2]);
    });
    it('[4x4] => [2x2]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(4, 4);
        expect(shape).toEqual([2, 2]);
    });
    it('[1024x1024] => [512x512]', function () {
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(1024, 1024);
        expect(shape).toEqual([512, 512]);
    });
    it('[MxN] => [ceil(N/2)xceil(M/2)]', function () {
        var M = 123;
        var N = 5013;
        var shape = tex_util.getPackedMatrixTextureShapeWidthHeight(M, N);
        expect(shape).toEqual([Math.ceil(N / 2), Math.ceil(M / 2)]);
    });
});
jasmine_util_1.describeWithFlags('tex_util encodeMatrixToUnpackedArray, channels = 4', test_util_1.WEBGL_ENVS, function () {
    it('1x1 writes the only matrix array value to the only texel', function () {
        var matrix = new Float32Array([1]);
        var unpackedRGBA = new Float32Array([0, 0, 0, 0]);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 4);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([1, 0, 0, 0]));
    });
    it('1x1 can upload texels with values greater than 1', function () {
        var matrix = new Float32Array([100]);
        var unpackedRGBA = new Float32Array([0, 0, 0, 0]);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 4);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([100, 0, 0, 0]));
    });
    it('1x4 each texel has 4 elements with matrix value in R channel', function () {
        var matrix = new Float32Array([1, 2, 3, 4]);
        var unpackedRGBA = new Float32Array(16);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 4);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]));
    });
});
jasmine_util_1.describeWithFlags('tex_util encodeMatrixToUnpackedArray, channels = 1', test_util_1.WEBGL_ENVS, function () {
    it('1x1 writes the only matrix array value to the only texel', function () {
        var matrix = new Float32Array([1]);
        var unpackedRGBA = new Float32Array([0]);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 1);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([1]));
    });
    it('1x1 can upload texels with values greater than 1', function () {
        var matrix = new Float32Array([100]);
        var unpackedRGBA = new Float32Array([0]);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 1);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([100]));
    });
    it('1x4 each texel has 4 elements with matrix value in R channel', function () {
        var matrix = new Float32Array([1, 2, 3, 4]);
        var unpackedRGBA = new Float32Array(4);
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedRGBA, 1);
        test_util_1.expectArraysClose(unpackedRGBA, new Float32Array([1, 2, 3, 4]));
    });
});
jasmine_util_1.describeWithFlags('tex_util decodeMatrixFromUnpackedArray', test_util_1.WEBGL_ENVS, function () {
    it('1x1 writes the only matrix array value to the first element', function () {
        var unpackedRGBA = new Float32Array([1, 0, 0, 0]);
        var matrix = new Float32Array(1);
        tex_util.decodeMatrixFromUnpackedArray(unpackedRGBA, matrix, 4);
        expect(matrix.length).toEqual(1);
        expect(matrix[0]).toEqual(1);
    });
    it('1x2 writes the second texel R component to the second element', function () {
        var unpackedRGBA = new Float32Array([1, 0, 0, 0, 2, 0, 0, 0]);
        var matrix = new Float32Array(2);
        tex_util.decodeMatrixFromUnpackedArray(unpackedRGBA, matrix, 4);
        expect(matrix.length).toEqual(2);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2]));
    });
});
jasmine_util_1.describeWithFlags('tex_util encodeMatrixToPackedRGBA', test_util_1.WEBGL_ENVS, function () {
    it('1x1 loads the element into R and 0\'s into GBA', function () {
        var matrix = new Float32Array([1]);
        var packedRGBA = new Float32Array(4);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 1, 1, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 0, 0, 0]));
    });
    it('1x2 loads the second element into G and 0\'s into BA', function () {
        var matrix = new Float32Array([1, 2]);
        var packedRGBA = new Float32Array(4);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 1, 2, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 2, 0, 0]));
    });
    it('2x1 loads the second element into G and 0\'s into BA', function () {
        var matrix = new Float32Array([1, 2]);
        var packedRGBA = new Float32Array(4);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 2, 1, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 0, 2, 0]));
    });
    it('2x2 exactly fills one texel', function () {
        var matrix = new Float32Array([1, 2, 3, 4]);
        var packedRGBA = new Float32Array(4);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 2, 2, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 2, 3, 4]));
    });
    it('4x3 pads the final column G and A channels with 0', function () {
        var matrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var packedRGBA = new Float32Array(16);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 4, 3, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11, 9, 0, 12, 0]));
    });
    it('3x4 pads the final row B and A channels with 0', function () {
        var matrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var packedRGBA = new Float32Array(16);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 3, 4, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0]));
    });
    it('3x3 bottom-right texel is R000', function () {
        var matrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var packedRGBA = new Float32Array(16);
        tex_util.encodeMatrixToPackedRGBA(matrix, 1, 3, 3, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0]));
    });
    it('2x3x4 texels in the last row of each batch are RG00', function () {
        var matrix = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]);
        var packedRGBA = new Float32Array(32);
        tex_util.encodeMatrixToPackedRGBA(matrix, 2, 3, 4, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([
            1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0,
            13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
        ]));
    });
    it('2x4x3 texels in the last column of each batch are R0B0', function () {
        var matrix = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]);
        var packedRGBA = new Float32Array(32);
        tex_util.encodeMatrixToPackedRGBA(matrix, 2, 4, 3, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([
            1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11, 9, 0, 12, 0,
            13, 14, 16, 17, 15, 0, 18, 0, 19, 20, 22, 23, 21, 0, 24, 0
        ]));
    });
    it('2x3x3 bottom right texel in each batch is R000', function () {
        var matrix = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);
        var packedRGBA = new Float32Array(32);
        tex_util.encodeMatrixToPackedRGBA(matrix, 2, 3, 3, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([
            1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0,
            10, 11, 13, 14, 12, 0, 15, 0, 16, 17, 0, 0, 18, 0, 0, 0
        ]));
    });
    it('4D (2x3x3x4) is properly encoded', function () {
        var matrix = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ]);
        var packedRGBA = new Float32Array(96);
        tex_util.encodeMatrixToPackedRGBA(matrix, 6, 3, 4, packedRGBA);
        test_util_1.expectArraysClose(packedRGBA, new Float32Array([
            1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0,
            13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0,
            25, 26, 29, 30, 27, 28, 31, 32, 33, 34, 0, 0, 35, 36, 0, 0,
            37, 38, 41, 42, 39, 40, 43, 44, 45, 46, 0, 0, 47, 48, 0, 0,
            49, 50, 53, 54, 51, 52, 55, 56, 57, 58, 0, 0, 59, 60, 0, 0,
            61, 62, 65, 66, 63, 64, 67, 68, 69, 70, 0, 0, 71, 72, 0, 0
        ]));
    });
});
jasmine_util_1.describeWithFlags('tex_util decodeMatrixFromPackedRGBA', test_util_1.WEBGL_ENVS, function () {
    it('1x1 matrix only loads R component from only texel', function () {
        var packedRGBA = new Float32Array([1, 0, 0, 0]);
        var matrix = new Float32Array(1);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 1, 1, matrix);
        expect(matrix[0]).toEqual(1);
    });
    it('1x2 matrix loads RG from only texel', function () {
        var packedRGBA = new Float32Array([1, 2, 0, 0]);
        var matrix = new Float32Array(2);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 1, 2, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2]));
    });
    it('2x1 matrix loads RB from only texel', function () {
        var packedRGBA = new Float32Array([1, 0, 2, 0]);
        var matrix = new Float32Array(2);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 2, 1, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2]));
    });
    it('2x2 matrix loads RGBA from only texel', function () {
        var packedRGBA = new Float32Array([1, 2, 3, 4]);
        var matrix = new Float32Array(4);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 2, 2, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2, 3, 4]));
    });
    it('4x3 final column only reads RB from edge texels', function () {
        var packedRGBA = new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11, 9, 0, 12, 0]);
        var matrix = new Float32Array(12);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 4, 3, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    });
    it('3x4 final row only reads RG from edge texels', function () {
        var packedRGBA = new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0]);
        var matrix = new Float32Array(12);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 3, 4, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    });
    it('3x3 bottom-right only reads R from corner texel', function () {
        var packedRGBA = new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0]);
        var matrix = new Float32Array(9);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 1, 3, 3, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    });
    it('2x3x4 bottom row in each batch only reads RG', function () {
        var packedRGBA = new Float32Array([
            1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0,
            13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
        ]);
        var matrix = new Float32Array(24);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 3, 4, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]));
    });
    it('2x4x3 final column in each batch only reads RB', function () {
        var packedRGBA = new Float32Array([
            1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 10, 11, 9, 0, 12, 0,
            13, 14, 16, 17, 15, 0, 18, 0, 19, 20, 22, 23, 21, 0, 24, 0
        ]);
        var matrix = new Float32Array(24);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 4, 3, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]));
    });
    it('2x3x3 bottom right texel in each batch only reads R', function () {
        var packedRGBA = new Float32Array([
            1, 2, 4, 5, 3, 0, 6, 0, 7, 8, 0, 0, 9, 0, 0, 0,
            10, 11, 13, 14, 12, 0, 15, 0, 16, 17, 0, 0, 18, 0, 0, 0
        ]);
        var matrix = new Float32Array(18);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 2, 3, 3, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]));
    });
    it('4D (2x3x3x4) is properly decoded', function () {
        var packedRGBA = new Float32Array([
            1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 0, 0, 11, 12, 0, 0,
            13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0,
            25, 26, 29, 30, 27, 28, 31, 32, 33, 34, 0, 0, 35, 36, 0, 0,
            37, 38, 41, 42, 39, 40, 43, 44, 45, 46, 0, 0, 47, 48, 0, 0,
            49, 50, 53, 54, 51, 52, 55, 56, 57, 58, 0, 0, 59, 60, 0, 0,
            61, 62, 65, 66, 63, 64, 67, 68, 69, 70, 0, 0, 71, 72, 0, 0
        ]);
        var matrix = new Float32Array(72);
        tex_util.decodeMatrixFromPackedRGBA(packedRGBA, 6, 3, 4, matrix);
        test_util_1.expectArraysClose(matrix, new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72
        ]));
    });
});
//# sourceMappingURL=tex_util_test.js.map