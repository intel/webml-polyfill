"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var broadcast_util = require("../../ops/broadcast_util");
var util = require("../../util");
var shader_util = require("./shader_compiler_util");
function makeShader(inputsInfo, outputShape, userCode, broadcast, usesPackedTextures) {
    var inputPrefixSnippet = inputsInfo.map(function (x) {
        var size = util.sizeFromShape(x.shapeInfo.logicalShape);
        if (x.shapeInfo.isUniform) {
            return "uniform float " + x.name + (size > 1 ? "[" + size + "]" : '') + ";";
        }
        return "uniform sampler2D " + x.name + ";";
    });
    inputPrefixSnippet = inputPrefixSnippet.join('\n');
    var inputSamplingSnippet = inputsInfo.map(function (x) { return getInputSamplingSnippet(x, outputShape, broadcast); })
        .join('\n');
    var outTexShape = outputShape.texShape;
    var outputSamplingSnippet;
    var floatTextureSetOutputSnippet;
    var shaderPrefix = SHADER_PREFIX;
    if (outputShape.isPacked) {
        outputSamplingSnippet =
            getPackedOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
        floatTextureSetOutputSnippet = FLOAT_TEXTURE_SET_RGBA_SNIPPET;
    }
    else {
        outputSamplingSnippet =
            getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
        floatTextureSetOutputSnippet = FLOAT_TEXTURE_SET_R_SNIPPET;
    }
    if (usesPackedTextures) {
        shaderPrefix += SHADER_PACKED_PREFIX;
    }
    var source = [
        shaderPrefix, FLOAT_TEXTURE_SAMPLE_SNIPPET, floatTextureSetOutputSnippet,
        inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet, userCode
    ].join('\n');
    return source;
}
exports.makeShader = makeShader;
function getSamplerFromInInfo(inInfo) {
    var shape = inInfo.shapeInfo.logicalShape;
    switch (shape.length) {
        case 0:
            return getSamplerScalar(inInfo);
        case 1:
            return getSampler1D(inInfo);
        case 2:
            return getSampler2D(inInfo);
        case 3:
            return getSampler3D(inInfo);
        case 4:
            return getSampler4D(inInfo);
        case 5:
            return getSampler5D(inInfo);
        case 6:
            return getSampler6D(inInfo);
        default:
            throw new Error(shape.length + "-D input sampling" +
                " is not yet supported");
    }
}
function getPackedSamplerFromInInfo(inInfo) {
    var shape = inInfo.shapeInfo.logicalShape;
    switch (shape.length) {
        case 1:
            return getPackedSampler1D(inInfo);
        case 2:
            return getPackedSampler2D(inInfo);
        case 3:
            return getPackedSampler3D(inInfo);
        case 4:
            return getPackedSampler4D(inInfo);
        default:
            throw new Error("Packed " + shape.length + "-D input sampling" +
                " is not yet supported");
    }
}
function getInputSamplingSnippet(inInfo, outShapeInfo, broadcast) {
    var res = getSamplerFlat(inInfo);
    if (inInfo.shapeInfo.isPacked) {
        res += getPackedSamplerFromInInfo(inInfo);
    }
    else {
        res += getSamplerFromInInfo(inInfo);
    }
    if (broadcast ||
        util.arraysEqual(inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
        res += getSamplerAtOutputCoords(inInfo, outShapeInfo, broadcast);
    }
    return res;
}
function getPackedOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 0:
            return getOutputScalarCoords();
        case 1:
            return getOutputPacked1DCoords(outShape, outTexShape);
        case 2:
            return getOutputPacked2DCoords(outShape, outTexShape);
        case 3:
            return getOutputPacked3DCoords(outShape, outTexShape);
        case 4:
            return getOutputPacked4DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D packed output " +
                "coordinate fetching is not yet supported");
    }
}
function getOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 0:
            return getOutputScalarCoords();
        case 1:
            return getOutput1DCoords(outShape, outTexShape);
        case 2:
            return getOutput2DCoords(outShape, outTexShape);
        case 3:
            return getOutput3DCoords(outShape, outTexShape);
        case 4:
            return getOutput4DCoords(outShape, outTexShape);
        case 5:
            return getOutput5DCoords(outShape, outTexShape);
        case 6:
            return getOutput6DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D output sampling is not yet supported");
    }
}
var SAMPLE_1D_SNIPPET = "\nvec2 UVfrom1D(int texNumR, int texNumC, int index) {\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom1D(int texNumR, int texNumC, int index) {\n  int texelIndex = index / 2;\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_2D_SNIPPET = "\nvec2 UVfrom2D(int texNumR, int texNumC, int numC, int row, int col) {\n  int index = row * numC + col;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,\n  int texNumC, int row, int col) {\n  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = texelIndex / texNumC;\n  int texC = texelIndex - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_3D_SNIPPET = "\nvec2 UVfrom3D(int texNumR, int texNumC, int stride0,\n    int stride1, int row, int col, int depth) {\n  // Explicitly use integer operations as dot() only works on floats.\n  int index = row * stride0 + col * stride1 + depth;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom3D(int texNumR, int texNumC,\n    int texelsInBatch, int texelsInLogicalRow, int b,\n    int row, int col) {\n  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_4D_SNIPPET = "\nvec2 UVfrom4D(int texNumR, int texNumC, int stride0,\n    int stride1, int stride2, int row, int col, int depth,\n    int depth2) {\n  // Explicitly use integer operations as dot() only works on floats.\n  int index = row * stride0 + col * stride1 + depth * stride2 + depth2;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\nvec2 packedUVfrom4D(int texNumR, int texNumC, int texelsInBatch2,\n    int texelsInBatch, int texelsInLogicalRow, int b2, int b,\n    int row, int col) {\n  int index = b2 * texelsInBatch2 + b * texelsInBatch +\n    (row / 2) * texelsInLogicalRow + (col / 2);\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_5D_SNIPPET = "\nvec2 UVfrom5D(int texNumR, int texNumC, int stride0,\n    int stride1, int stride2, int stride3, int row, int col, int depth,\n    int depth2, int depth3) {\n  // Explicitly use integer operations as dot() only works on floats.\n  int index = row * stride0 + col * stride1 +\n              depth * stride2 + depth2 * stride3 + depth3;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_6D_SNIPPET = "\nvec2 UVfrom6D(int texNumR, int texNumC, int stride0,\n    int stride1, int stride2, int stride3, int stride4,\n    int row, int col, int depth, int depth2, int depth3, int depth4) {\n  // Explicitly use integer operations as dot() only works on floats.\n  int index = row * stride0 + col * stride1 + depth * stride2 + depth2 *\n    stride3 + depth3 * stride4 + depth4;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var FLOAT_TEXTURE_SAMPLE_SNIPPET = "\n  float sampleTexture(sampler2D textureSampler, vec2 uv) {\n    return texture(textureSampler, uv).r;\n  }\n";
var FLOAT_TEXTURE_SET_R_SNIPPET = "\n  void setOutput(float val) {\n    outputColor = vec4(val, 0, 0, 0);\n  }\n";
var FLOAT_TEXTURE_SET_RGBA_SNIPPET = "\n  void setOutput(vec4 val) {\n    outputColor = val;\n  }\n";
var SHADER_PREFIX = "#version 300 es\n  precision highp float;\n  precision highp int;\n  precision highp sampler2D;\n\n  in vec2 resultUV;\n  out vec4 outputColor;\n  const vec2 halfCR = vec2(0.5, 0.5);\n\n  struct ivec5\n  {\n    int x;\n    int y;\n    int z;\n    int w;\n    int u;\n  };\n\n  struct ivec6\n  {\n    int x;\n    int y;\n    int z;\n    int w;\n    int u;\n    int v;\n  };\n\n  bool isNaN(float val) {\n    return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;\n  }\n\n  bool hasNaN(vec4 values) {\n    vec4 v1 = values * values;\n    vec4 v2 = values * values;\n    return any(notEqual(v1, v2));\n  }\n\n  float getNaN(vec4 values) {\n    return dot(vec4(1), values);\n  }\n\n  // int round(float value) {\n  //   return int(floor(value + 0.5));\n  // }\n\n  int imod(int x, int y) {\n    return x - y * (x / y);\n  }\n\n  //Based on the work of Dave Hoskins\n  //https://www.shadertoy.com/view/4djSRW\n  #define HASHSCALE1 443.8975\n  float random(float seed){\n    vec2 p = resultUV * seed;\n    vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);\n    p3 += dot(p3, p3.yzx + 19.19);\n    return fract((p3.x + p3.y) * p3.z);\n  }\n\n  " + SAMPLE_1D_SNIPPET + "\n  " + SAMPLE_2D_SNIPPET + "\n  " + SAMPLE_3D_SNIPPET + "\n  " + SAMPLE_4D_SNIPPET + "\n  " + SAMPLE_5D_SNIPPET + "\n  " + SAMPLE_6D_SNIPPET + "\n";
var SHADER_PACKED_PREFIX = "\n  float getChannel(vec4 frag, vec2 innerDims) {\n    vec2 modCoord = mod(innerDims, 2.);\n    return modCoord.x == 0. ?\n      (modCoord.y == 0. ? frag.r : frag.g) :\n      (modCoord.y == 0. ? frag.b : frag.a);\n  }\n  float getChannel(vec4 frag, int dim) {\n    float modCoord = mod(float(dim), 2.);\n    return modCoord == 0. ? frag.r : frag.g;\n  }\n";
function getOutputScalarCoords() {
    return "\n    int getOutputCoords() {\n      return 0;\n    }\n  ";
}
function getOutputPacked1DCoords(shape, texShape) {
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (texShape[0] === 1) {
        return "\n      int getOutputCoords() {\n        return 2 * int(resultUV.x * " + packedTexShape[1] + ".0);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      int getOutputCoords() {\n        return 2 * int(resultUV.y * " + packedTexShape[0] + ".0);\n      }\n    ";
    }
    return "\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      return resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n    }\n  ";
}
function getOutput1DCoords(shape, texShape) {
    if (texShape[0] === 1) {
        return "\n      int getOutputCoords() {\n        return int(resultUV.x * " + texShape[1] + ".0);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      int getOutputCoords() {\n        return int(resultUV.y * " + texShape[0] + ".0);\n      }\n    ";
    }
    return "\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      return resTexRC.x * " + texShape[1] + " + resTexRC.y;\n    }\n  ";
}
function getOutputPacked3DCoords(shape, texShape) {
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    var texelsInLogicalRow = Math.ceil(shape[2] / 2);
    var texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);
    return "\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n\n      int b = index / " + texelsInBatch + ";\n      index -= b * " + texelsInBatch + ";\n\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec3(b, r, c);\n    }\n  ";
}
function getOutput3DCoords(shape, texShape) {
    var coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
    return "\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      " + coordsFromIndexSnippet + "\n      return ivec3(r, c, d);\n    }\n  ";
}
function getOutputPacked4DCoords(shape, texShape) {
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    var texelsInLogicalRow = Math.ceil(shape[3] / 2);
    var texelsInBatch = texelsInLogicalRow * Math.ceil(shape[2] / 2);
    var texelsInBatch2 = texelsInBatch * shape[1];
    return "\n    ivec4 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n\n      int b2 = index / " + texelsInBatch2 + ";\n      index -= b2 * " + texelsInBatch2 + ";\n\n      int b = index / " + texelsInBatch + ";\n      index -= b * " + texelsInBatch + ";\n\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec4(b2, b, r, c);\n    }\n  ";
}
function getOutput4DCoords(shape, texShape) {
    var coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2'], shape);
    return "\n    ivec4 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      " + coordsFromIndexSnippet + "\n      return ivec4(r, c, d, d2);\n    }\n  ";
}
function getOutput5DCoords(shape, texShape) {
    var coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3'], shape);
    return "\n    ivec5 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx * vec2(" + texShape[0] + ",\n                             " + texShape[1] + "));\n\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n\n      " + coordsFromIndexSnippet + "\n\n      ivec5 outShape = ivec5(r, c, d, d2, d3);\n      return outShape;\n    }\n  ";
}
function getOutput6DCoords(shape, texShape) {
    var coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);
    return "\n    ivec6 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n        vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n\n      " + coordsFromIndexSnippet + "\n\n      ivec6 result = ivec6(r, c, d, d2, d3, d4);\n      return result;\n    }\n  ";
}
function getOutputPacked2DCoords(shape, texShape) {
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (util.arraysEqual(shape, texShape)) {
        return "\n      ivec2 getOutputCoords() {\n        return 2 * ivec2(resultUV.yx * vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n      }\n    ";
    }
    var texelsInLogicalRow = Math.ceil(shape[1] / 2);
    return "\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + packedTexShape[0] + ", " + packedTexShape[1] + "));\n\n      int index = resTexRC.x * " + packedTexShape[1] + " + resTexRC.y;\n      int r = 2 * (index / " + texelsInLogicalRow + ");\n      int c = imod(index, " + texelsInLogicalRow + ") * 2;\n\n      return ivec2(r, c);\n    }\n  ";
}
function getOutput2DCoords(shape, texShape) {
    if (util.arraysEqual(shape, texShape)) {
        return "\n      ivec2 getOutputCoords() {\n        return ivec2(resultUV.yx * vec2(" + texShape[0] + ", " + texShape[1] + "));\n      }\n    ";
    }
    if (shape[1] === 1) {
        return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(index, 0);\n      }\n    ";
    }
    if (shape[0] === 1) {
        return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                               vec2(" + texShape[0] + ", " + texShape[1] + "));\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(0, index);\n      }\n    ";
    }
    return "\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + texShape[0] + ", " + texShape[1] + "));\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      int r = index / " + shape[1] + ";\n      int c = index - r * " + shape[1] + ";\n      return ivec2(r, c);\n    }\n  ";
}
function getSamplerScalar(inputInfo) {
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    if (inputInfo.shapeInfo.isUniform) {
        return "float " + funcName + "() {return " + texName + ";}";
    }
    return "\n    float " + funcName + "() {\n      return sampleTexture(" + texName + ", halfCR);\n    }\n  ";
}
function getPackedSampler1D(inputInfo) {
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var texShape = inputInfo.shapeInfo.texShape;
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    return "\n    vec4 " + funcName + "(int index) {\n      vec2 uv = packedUVfrom1D(\n        " + packedTexShape[0] + ", " + packedTexShape[1] + ", index);\n      return texture(" + texName + ", uv);\n    }\n  ";
}
function getSampler1D(inputInfo) {
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    return "\n    float " + funcName + "(int index) {\n      return " + funcName + "Flat(index);\n    }\n  ";
}
function getPackedSampler2D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var texShape = inputInfo.shapeInfo.texShape;
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texShape != null && util.arraysEqual(shape, texShape)) {
        return "\n      vec4 " + funcName + "(int row, int col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + texNumC + ".0, " + texNumR + ".0);\n\n        return texture(" + texName + ", uv);\n      }\n    ";
    }
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    var valuesPerRow = Math.ceil(shape[1] / 2);
    return "\n    vec4 " + funcName + "(int row, int col) {\n      vec2 uv = packedUVfrom2D(" + valuesPerRow + ", " + packedTexShape[0] + ", " + packedTexShape[1] + ", row, col);\n      return texture(" + texName + ", uv);\n    }\n  ";
}
function getSampler2D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var texShape = inputInfo.shapeInfo.texShape;
    if (texShape != null && util.arraysEqual(shape, texShape)) {
        var texNumR_1 = texShape[0];
        var texNumC_1 = texShape[1];
        return "\n    float " + funcName + "(int row, int col) {\n      vec2 uv = (vec2(col, row) + halfCR) / vec2(" + texNumC_1 + ".0, " + texNumR_1 + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    var _a = util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
    var squeezedShape = newShape;
    if (squeezedShape.length < shape.length) {
        var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        var params = ['row', 'col'];
        return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
    }
    if (inputInfo.shapeInfo.isUniform) {
        return "\n      float " + funcName + "(int row, int col) {\n        float index = dot(vec2(row, col), vec2(" + shape[1] + ", 1));\n        return " + funcName + "Flat(int(round(index)));\n      }\n    ";
    }
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texNumC === 1) {
        return "\n    float " + funcName + "(int row, int col) {\n      float index = dot(vec2(row, col), vec2(" + shape[1] + ", 1));\n      vec2 uv = vec2(0.5, (index + 0.5) / " + texNumR + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    if (texNumR === 1) {
        return "\n    float " + funcName + "(int row, int col) {\n      float index = dot(vec2(row, col), vec2(" + shape[1] + ", 1));\n      vec2 uv = vec2((index + 0.5) / " + texNumC + ".0, 0.5);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    return "\n  float " + funcName + "(int row, int col) {\n    vec2 uv = UVfrom2D(" + texNumR + ", " + texNumC + ", " + shape[1] + ", row, col);\n    return sampleTexture(" + texName + ", uv);\n  }\n";
}
function getPackedSampler3D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var texShape = inputInfo.shapeInfo.texShape;
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (shape[0] === 1) {
        var squeezedShape = shape.slice(1);
        var keptDims = [1, 2];
        var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        var params = ['b', 'row', 'col'];
        return "\n        " + getPackedSamplerFromInInfo(newInputInfo) + "\n        vec4 " + funcName + "(int b, int row, int col) {\n          return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n        }\n      ";
    }
    var texNumR = packedTexShape[0];
    var texNumC = packedTexShape[1];
    var valuesPerRow = Math.ceil(shape[2] / 2);
    var texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);
    return "\n    vec4 " + funcName + "(int b, int row, int col) {\n      vec2 uv = packedUVfrom3D(\n        " + texNumR + ", " + texNumC + ", " + texelsInBatch + ", " + valuesPerRow + ", b, row, col);\n      return texture(" + texName + ", uv);\n    }\n  ";
}
function getSampler3D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var stride0 = shape[1] * shape[2];
    var stride1 = shape[2];
    var _a = util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
    var squeezedShape = newShape;
    if (squeezedShape.length < shape.length) {
        var newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        var params = ['row', 'col', 'depth'];
        return "\n        " + getSamplerFromInInfo(newInputInfo) + "\n        float " + funcName + "(int row, int col, int depth) {\n          return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n        }\n      ";
    }
    if (inputInfo.shapeInfo.isUniform) {
        return "\n      float " + funcName + "(int row, int col, int depth) {\n        float index = dot(vec3(row, col, depth),\n                          vec3(" + stride0 + ", " + stride1 + ", 1));\n        return " + funcName + "Flat(int(round(index)));\n      }\n    ";
    }
    var texShape = inputInfo.shapeInfo.texShape;
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texNumC === stride0) {
        return "\n        float " + funcName + "(int row, int col, int depth) {\n          float texR = float(row);\n          float texC = dot(vec2(col, depth), vec2(" + stride1 + ", 1));\n          vec2 uv = (vec2(texC, texR) + halfCR) /\n                     vec2(" + texNumC + ".0, " + texNumR + ".0);\n          return sampleTexture(" + texName + ", uv);\n        }\n      ";
    }
    if (texNumC === stride1) {
        return "\n    float " + funcName + "(int row, int col, int depth) {\n      float texR = dot(vec2(row, col), vec2(" + shape[1] + ", 1));\n      float texC = float(depth);\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + texNumC + ".0, " + texNumR + ".0);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
    }
    return "\n      float " + funcName + "(int row, int col, int depth) {\n        vec2 uv = UVfrom3D(\n            " + texNumR + ", " + texNumC + ", " + stride0 + ", " + stride1 + ", row, col, depth);\n        return sampleTexture(" + texName + ", uv);\n      }\n  ";
}
function getPackedSampler4D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var texShape = inputInfo.shapeInfo.texShape;
    var packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    var texNumR = packedTexShape[0];
    var texNumC = packedTexShape[1];
    var valuesPerRow = Math.ceil(shape[3] / 2);
    var texelsInBatch = valuesPerRow * Math.ceil(shape[2] / 2);
    var texelsInBatch2 = texelsInBatch * shape[1];
    return "\n    vec4 " + funcName + "(int b2, int b, int row, int col) {\n      vec2 uv = packedUVfrom4D(\n        " + texNumR + ", " + texNumC + ", " + texelsInBatch2 + ",\n        " + texelsInBatch + ", " + valuesPerRow + ", b2, b, row, col);\n      return texture(" + texName + ", uv);\n    }\n  ";
}
function getSampler4D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var stride2 = shape[3];
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    var _a = util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
    if (newShape.length < shape.length) {
        var newInputInfo = squeezeInputInfo(inputInfo, newShape);
        var params = ['row', 'col', 'depth', 'depth2'];
        return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
    }
    if (inputInfo.shapeInfo.isUniform) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        float index = dot(vec4(row, col, depth, depth2),\n                          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", 1));\n        return " + funcName + "Flat(int(round(index)));\n      }\n    ";
    }
    var texShape = inputInfo.shapeInfo.texShape;
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texNumC === stride0) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        float texR = float(row);\n        float texC =\n            dot(vec3(col, depth, depth2), vec3(" + stride1 + ", " + stride2 + ", 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    if (texNumC === stride2) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        float texR = dot(vec3(row, col, depth),\n                         vec3(" + shape[1] * shape[2] + ", " + shape[2] + ", 1));\n        float texC = float(depth2);\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col, int depth, int depth2) {\n      vec2 uv = UVfrom4D(" + texNumR + ", " + texNumC + ", " + stride0 + ", " + stride1 + ",\n          " + stride2 + ", row, col, depth, depth2);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
}
function getSampler5D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var stride3 = shape[4];
    var stride2 = shape[3] * stride3;
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    var _a = util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
    if (newShape.length < shape.length) {
        var newInputInfo = squeezeInputInfo(inputInfo, newShape);
        var params = ['row', 'col', 'depth', 'depth2', 'depth3'];
        return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
    }
    if (inputInfo.shapeInfo.isUniform) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        float index = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", " + stride3 + ")) +\n          depth3;\n        return " + funcName + "Flat(index);\n      }\n    ";
    }
    var texShape = inputInfo.shapeInfo.texShape;
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texNumC === stride0) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        int texR = row;\n        float texC = dot(\n          vec4(col, depth, depth2, depth3),\n          vec4(" + stride1 + ", " + stride2 + ", " + stride3 + ", 1));\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    if (texNumC === stride3) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n        float texR = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + shape[1] * shape[2] * shape[3] + ", " + shape[2] * shape[3] + ",\n            " + shape[3] + ", 1));\n        int texC = depth3;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col, int depth, int depth2, int depth3) {\n      vec2 uv = UVfrom5D(" + texNumR + ", " + texNumC + ", " + stride0 + ", " + stride1 + ",\n          " + stride2 + ", " + stride3 + ", row, col, depth, depth2, depth3);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
}
function getSampler6D(inputInfo) {
    var shape = inputInfo.shapeInfo.logicalShape;
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var stride4 = shape[5];
    var stride3 = shape[4] * stride4;
    var stride2 = shape[3] * stride3;
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    var _a = util.squeezeShape(shape), newShape = _a.newShape, keptDims = _a.keptDims;
    if (newShape.length < shape.length) {
        var newInputInfo = squeezeInputInfo(inputInfo, newShape);
        var params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
        return "\n      " + getSamplerFromInInfo(newInputInfo) + "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        return " + funcName + "(" + getSqueezedParams(params, keptDims) + ");\n      }\n    ";
    }
    if (inputInfo.shapeInfo.isUniform) {
        return "\n      float " + funcName + "(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n        float index = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + stride0 + ", " + stride1 + ", " + stride2 + ", " + stride3 + ")) +\n          dot(\n            vec2(depth3, depth4),\n            vec2(" + stride4 + ", 1));\n        return " + funcName + "Flat(index);\n      }\n    ";
    }
    var texShape = inputInfo.shapeInfo.texShape;
    var texNumR = texShape[0];
    var texNumC = texShape[1];
    if (texNumC === stride0) {
        return "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        int texR = row;\n        float texC = dot(\n          vec4(col, depth, depth2, depth3),\n          vec4(" + stride1 + ", " + stride2 + ", " + stride3 + ", " + stride4 + ")) + depth4;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                   vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    if (texNumC === stride4) {
        return "\n      float " + funcName + "(int row, int col, int depth,\n                    int depth2, int depth3, int depth4) {\n        float texR = dot(\n          vec4(row, col, depth, depth2),\n          vec4(" + shape[1] * shape[2] * shape[3] * shape[4] + ",\n               " + shape[2] * shape[3] * shape[4] + ",\n               " + shape[3] * shape[4] + ",\n               " + shape[4] + ")) + depth3;\n        int texC = depth4;\n        vec2 uv = (vec2(texC, texR) + halfCR) /\n                  vec2(" + texNumC + ".0, " + texNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col, int depth,\n                  int depth2, int depth3, int depth4) {\n      vec2 uv = UVfrom6D(" + texNumR + ", " + texNumC + ", " + stride0 + ", " + stride1 + ",\n          " + stride2 + ", " + stride3 + ", " + stride4 + "\n          ,row, col, depth, depth2, depth3, depth4);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
}
function getSamplerFlat(inputInfo) {
    var texName = inputInfo.name;
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
    var inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
    if (inputInfo.shapeInfo.isUniform) {
        if (inSize === 1) {
            return "float " + funcName + "(int index) {return " + texName + ";}";
        }
        return "\n      float " + funcName + "(int index) {\n        for (int i = 0; i < " + inSize + "; i++) {\n          if (i == index) {\n            return " + texName + "[i];\n          }\n        }\n      }\n    ";
    }
    var texShape = inputInfo.shapeInfo.texShape;
    var tNumR = texShape[0];
    var tNumC = texShape[1];
    if (tNumC === 1 && tNumR === 1) {
        return "\n      float " + funcName + "(int index) {\n        return sampleTexture(" + texName + ", halfCR);\n      }\n    ";
    }
    if (tNumC === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2(0.5, (float(index) + 0.5) / " + tNumR + ".0);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    if (tNumR === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2((float(index) + 0.5) / " + tNumC + ".0, 0.5);\n        return sampleTexture(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int index) {\n      vec2 uv = UVfrom1D(" + tNumR + ", " + tNumC + ", index);\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
}
function getBroadcastOutputCoordsSampler(inputInfo, outShapeInfo, texFuncSnippet, funcName) {
    var inRank = inputInfo.shapeInfo.logicalShape.length;
    var outRank = outShapeInfo.logicalShape.length;
    var type = 'int';
    if (outRank === 2) {
        type = 'ivec2';
    }
    else if (outRank === 3) {
        type = 'ivec3';
    }
    else if (outRank === 4) {
        type = 'ivec4';
    }
    var broadcastDims = broadcast_util.getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
    var rankDiff = outRank - inRank;
    var coordsSnippet;
    if (inRank === 0) {
        coordsSnippet = '';
    }
    else if (outRank < 2 && broadcastDims.length >= 1) {
        coordsSnippet = 'coords = 0;';
    }
    else {
        coordsSnippet =
            broadcastDims.map(function (d) { return "coords[" + (d + rankDiff) + "] = 0;"; }).join('\n');
    }
    var unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
            .map(function (s, i) { return "coords[" + (i + rankDiff) + "]"; })
            .join(', ');
    }
    return "\n    float " + funcName + "() {\n      " + type + " coords = getOutputCoords();\n      " + coordsSnippet + "\n      return get" + texFuncSnippet + "(" + unpackedCoordsSnippet + ");\n    }\n  ";
}
function getSamplerAtOutputCoords(inputInfo, outShapeInfo, supportsBroadcasting) {
    var texName = inputInfo.name;
    var texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    var funcName = 'get' + texFuncSnippet + 'AtOutCoords';
    var broadcastDims = broadcast_util.getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
    var inRank = inputInfo.shapeInfo.logicalShape.length;
    var outRank = outShapeInfo.logicalShape.length;
    var doBroadcast = supportsBroadcasting && ((outRank > inRank) || broadcastDims.length > 0);
    var broadcastOverOuter = broadcast_util.broadcastDimsAreOuter(broadcastDims);
    var isUniform = inputInfo.shapeInfo.isUniform;
    if (doBroadcast && !broadcastOverOuter) {
        return getBroadcastOutputCoordsSampler(inputInfo, outShapeInfo, texFuncSnippet, funcName);
    }
    var inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
    var broadcastSnippet = '';
    if (doBroadcast && broadcastOverOuter) {
        broadcastSnippet = "\n        int mainPart = index / " + inSize + ";\n        index -= mainPart * " + inSize + ";\n      ";
    }
    var outTexShape = outShapeInfo.texShape;
    if (isUniform) {
        if (inSize === 1) {
            return "float " + funcName + "() {return " + texName + ";}";
        }
        return "\n      float " + funcName + "() {\n        ivec2 resTexRC = ivec2(resultUV.yx *\n                              vec2(" + outTexShape[0] + ", " + outTexShape[1] + "));\n        int index = resTexRC.x * " + outTexShape[1] + " + resTexRC.y;\n        " + broadcastSnippet + "\n        return get" + texFuncSnippet + "Flat(index);\n      }\n    ";
    }
    var inTexShape = inputInfo.shapeInfo.texShape;
    if (util.arraysEqual(inTexShape, outTexShape)) {
        return "\n      float " + funcName + "() {\n        return sampleTexture(" + texName + ", resultUV);\n      }\n    ";
    }
    return "\n    float " + funcName + "() {\n      ivec2 resTexRC = ivec2(resultUV.yx *\n                             vec2(" + outTexShape[0] + ", " + outTexShape[1] + "));\n      int index = resTexRC.x * " + outTexShape[1] + " + resTexRC.y;\n      " + broadcastSnippet + "\n      int texR = index / " + inTexShape[1] + ";\n      int texC = index - texR * " + inTexShape[1] + ";\n      vec2 uv = (vec2(texC, texR) + halfCR) /\n                 vec2(" + inTexShape[1] + ".0, " + inTexShape[0] + ".0);\n\n      return sampleTexture(" + texName + ", uv);\n    }\n  ";
}
function getCoordsDataType(rank) {
    if (rank <= 1) {
        return 'int';
    }
    else if (rank === 2) {
        return 'ivec2';
    }
    else if (rank === 3) {
        return 'ivec3';
    }
    else if (rank === 4) {
        return 'ivec4';
    }
    else if (rank === 5) {
        return 'ivec5';
    }
    else if (rank === 6) {
        return 'ivec6';
    }
    else {
        throw Error("GPU for rank " + rank + " is not yet supported");
    }
}
exports.getCoordsDataType = getCoordsDataType;
function squeezeInputInfo(inInfo, squeezedShape) {
    var newInputInfo = JSON.parse(JSON.stringify(inInfo));
    newInputInfo.shapeInfo.logicalShape = squeezedShape;
    return newInputInfo;
}
function getSqueezedParams(params, keptDims) {
    return keptDims.map(function (d) { return params[d]; }).join(', ');
}
//# sourceMappingURL=shader_compiler.js.map