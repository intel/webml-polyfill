"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var TextureUsage;
(function (TextureUsage) {
    TextureUsage[TextureUsage["RENDER"] = 0] = "RENDER";
    TextureUsage[TextureUsage["UPLOAD"] = 1] = "UPLOAD";
    TextureUsage[TextureUsage["PIXELS"] = 2] = "PIXELS";
    TextureUsage[TextureUsage["DOWNLOAD"] = 3] = "DOWNLOAD";
})(TextureUsage = exports.TextureUsage || (exports.TextureUsage = {}));
var PhysicalTextureType;
(function (PhysicalTextureType) {
    PhysicalTextureType[PhysicalTextureType["UNPACKED_FLOAT16"] = 0] = "UNPACKED_FLOAT16";
    PhysicalTextureType[PhysicalTextureType["UNPACKED_FLOAT32"] = 1] = "UNPACKED_FLOAT32";
    PhysicalTextureType[PhysicalTextureType["PACKED_4X1_UNSIGNED_BYTE"] = 2] = "PACKED_4X1_UNSIGNED_BYTE";
    PhysicalTextureType[PhysicalTextureType["PACKED_2X2_FLOAT32"] = 3] = "PACKED_2X2_FLOAT32";
    PhysicalTextureType[PhysicalTextureType["PACKED_2X2_FLOAT16"] = 4] = "PACKED_2X2_FLOAT16";
})(PhysicalTextureType = exports.PhysicalTextureType || (exports.PhysicalTextureType = {}));
function getUnpackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns, rows];
}
exports.getUnpackedMatrixTextureShapeWidthHeight = getUnpackedMatrixTextureShapeWidthHeight;
function getUnpackedArraySizeFromMatrixSize(matrixSize, channelsPerTexture) {
    return matrixSize * channelsPerTexture;
}
exports.getUnpackedArraySizeFromMatrixSize = getUnpackedArraySizeFromMatrixSize;
function getColorMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns * 4, rows];
}
exports.getColorMatrixTextureShapeWidthHeight = getColorMatrixTextureShapeWidthHeight;
function getMatrixSizeFromUnpackedArraySize(unpackedSize, channelsPerTexture) {
    if (unpackedSize % channelsPerTexture !== 0) {
        throw new Error("unpackedSize (" + unpackedSize + ") must be a multiple of " +
            ("" + channelsPerTexture));
    }
    return unpackedSize / channelsPerTexture;
}
exports.getMatrixSizeFromUnpackedArraySize = getMatrixSizeFromUnpackedArraySize;
function encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture) {
    var requiredSize = getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture);
    if (unpackedArray.length < requiredSize) {
        throw new Error("unpackedArray length (" + unpackedArray.length + ") must be >= " +
            ("" + requiredSize));
    }
    var dst = 0;
    for (var src = 0; src < matrix.length; ++src) {
        unpackedArray[dst] = matrix[src];
        dst += channelsPerTexture;
    }
}
exports.encodeMatrixToUnpackedArray = encodeMatrixToUnpackedArray;
function decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture) {
    var requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
    if (matrix.length < requiredSize) {
        throw new Error("matrix length (" + matrix.length + ") must be >= " + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < unpackedArray.length; src += channelsPerTexture) {
        matrix[dst++] = unpackedArray[src];
    }
}
exports.decodeMatrixFromUnpackedArray = decodeMatrixFromUnpackedArray;
function decodeMatrixFromUnpackedColorRGBAArray(unpackedArray, matrix, channels) {
    var requiredSize = unpackedArray.length * channels / 4;
    if (matrix.length < requiredSize) {
        throw new Error("matrix length (" + matrix.length + ") must be >= " + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < unpackedArray.length; src += 4) {
        for (var c = 0; c < channels; c++) {
            matrix[dst++] = unpackedArray[src + c];
        }
    }
}
exports.decodeMatrixFromUnpackedColorRGBAArray = decodeMatrixFromUnpackedColorRGBAArray;
function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [Math.ceil(columns / 2), Math.ceil(rows / 2)];
}
exports.getPackedMatrixTextureShapeWidthHeight = getPackedMatrixTextureShapeWidthHeight;
function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    return w * h * 4;
}
exports.getPackedRGBAArraySizeFromMatrixShape = getPackedRGBAArraySizeFromMatrixShape;
function encodeMatrixToPackedRGBA(matrix, batches, rows, columns, packedRGBA) {
    var requiredSize = getPackedRGBAArraySizeFromMatrixShape(rows, columns);
    if (packedRGBA.length < requiredSize) {
        throw new Error("packedRGBA length (" + packedRGBA.length + ") must be >=\n        " + requiredSize);
    }
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    var texelsPerRow = Math.ceil(columns / 2);
    var texelsPerBatch = texelsPerRow * Math.ceil(rows / 2);
    var flattenedMatrixSize = util.nearestLargerEven(rows) * util.nearestLargerEven(columns);
    for (var batch = 0; batch < batches; batch++) {
        var sourceOffset = batch * rows * columns;
        var batchOffset = batch * flattenedMatrixSize;
        {
            var dstStride = (oddWidth ? 4 : 0);
            var oneRow = columns;
            var dst = batchOffset;
            for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
                var matrixSrcRow = (blockY * 2 * columns);
                for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                    var matrixSrcCol = blockX * 2;
                    var src = sourceOffset + matrixSrcRow + matrixSrcCol;
                    packedRGBA[dst] = matrix[src];
                    packedRGBA[dst + 1] = matrix[src + 1];
                    packedRGBA[dst + 2] = matrix[src + oneRow];
                    packedRGBA[dst + 3] = matrix[src + oneRow + 1];
                    dst += 4;
                }
                dst += dstStride;
            }
        }
        if (oddWidth) {
            var src = sourceOffset + columns - 1;
            var dst = batchOffset + (texelsPerRow - 1) * 4;
            var srcStride = 2 * columns;
            var dstStride = texelsPerRow * 4;
            for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
                packedRGBA[dst] = matrix[src];
                packedRGBA[dst + 2] = matrix[src + columns];
                src += srcStride;
                dst += dstStride;
            }
        }
        if (oddHeight) {
            var src = sourceOffset + (rows - 1) * columns;
            var dst = batchOffset + (texelsPerBatch - texelsPerRow) * 4;
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                packedRGBA[dst++] = matrix[src++];
                packedRGBA[dst++] = matrix[src++];
                dst += 2;
            }
            if (oddWidth && oddHeight) {
                packedRGBA[batchOffset + flattenedMatrixSize - 4] = matrix[src];
            }
        }
    }
    return packedRGBA;
}
exports.encodeMatrixToPackedRGBA = encodeMatrixToPackedRGBA;
function decodeMatrixFromPackedRGBA(packedRGBA, batches, rows, columns, matrix) {
    var requiredSize = rows * columns;
    if (matrix.length < requiredSize) {
        throw new Error("matrix length (" + matrix.length + ") must be >= " + requiredSize);
    }
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    var texelsPerRow = Math.ceil(columns / 2);
    var texelsPerBatch = texelsPerRow * Math.ceil(rows / 2);
    var flattenedMatrixSize = util.nearestLargerEven(rows) * util.nearestLargerEven(columns);
    for (var batch = 0; batch < batches; batch++) {
        var batchOffset = batch * rows * columns;
        var sourceOffset = batch * flattenedMatrixSize;
        {
            var srcStride = oddWidth ? 4 : 0;
            var dstStride = columns + (oddWidth ? 1 : 0);
            var src = sourceOffset;
            var dstRow1 = batchOffset;
            var dstRow2 = batchOffset + columns;
            for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
                for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                    matrix[dstRow1++] = packedRGBA[src++];
                    matrix[dstRow1++] = packedRGBA[src++];
                    matrix[dstRow2++] = packedRGBA[src++];
                    matrix[dstRow2++] = packedRGBA[src++];
                }
                src += srcStride;
                dstRow1 += dstStride;
                dstRow2 += dstStride;
            }
        }
        if (oddWidth) {
            var src = sourceOffset + (texelsPerRow - 1) * 4;
            var dst = batchOffset + columns - 1;
            var srcStride = texelsPerRow * 4;
            var dstStride = 2 * columns;
            for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
                matrix[dst] = packedRGBA[src];
                matrix[dst + columns] = packedRGBA[src + 2];
                src += srcStride;
                dst += dstStride;
            }
        }
        if (oddHeight) {
            var src = sourceOffset + (texelsPerBatch - texelsPerRow) * 4;
            var dst = batchOffset + (rows - 1) * columns;
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                matrix[dst++] = packedRGBA[src++];
                matrix[dst++] = packedRGBA[src++];
                src += 2;
            }
            if (oddWidth) {
                matrix[batchOffset + (rows * columns) - 1] = packedRGBA[src];
            }
        }
    }
    return matrix;
}
exports.decodeMatrixFromPackedRGBA = decodeMatrixFromPackedRGBA;
//# sourceMappingURL=tex_util.js.map