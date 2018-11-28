"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var shader_compiler = require("./shader_compiler");
function compileProgram(gpgpu, program, inputs, output) {
    var userCode = program.userCode;
    var inputInfos = inputs.map(function (input, i) {
        var shapeInfo = {
            logicalShape: input.shape,
            texShape: input.isUniform ? null : input.texData.texShape,
            isUniform: input.isUniform,
            isPacked: input.isUniform ? false : input.texData.isPacked
        };
        return { name: program.variableNames[i], shapeInfo: shapeInfo };
    });
    var inShapeInfos = inputInfos.map(function (x) { return x.shapeInfo; });
    var outShapeInfo = {
        logicalShape: output.shape,
        texShape: output.texData.texShape,
        isUniform: false,
        isPacked: output.texData.isPacked
    };
    var source = shader_compiler.makeShader(inputInfos, outShapeInfo, userCode, program.supportsBroadcasting === true, program.usesPackedTextures);
    var webGLProgram = gpgpu.createProgram(source);
    var uniformLocations = {};
    for (var i = 0; i < program.variableNames.length; i++) {
        var uniformName = program.variableNames[i];
        var shouldThrow = false;
        uniformLocations[uniformName] =
            gpgpu.getUniformLocation(webGLProgram, uniformName, shouldThrow);
    }
    return {
        program: program,
        source: source,
        webGLProgram: webGLProgram,
        uniformLocations: uniformLocations,
        gpgpu: gpgpu,
        inShapeInfos: inShapeInfos,
        outShapeInfo: outShapeInfo
    };
}
exports.compileProgram = compileProgram;
function validateBinaryAndProgram(shapeInfos, inputs) {
    if (shapeInfos.length !== inputs.length) {
        throw Error("Binary was compiled with " + shapeInfos.length + " inputs, but " +
            ("was executed with " + inputs.length + " inputs"));
    }
    shapeInfos.forEach(function (s, i) {
        var shapeA = s.logicalShape;
        var input = inputs[i];
        var shapeB = input.shape;
        if (!util.arraysEqual(shapeA, shapeB)) {
            throw Error("Binary was compiled with different shapes than " +
                ("the current args. Shapes " + shapeA + " and " + shapeB + " must match"));
        }
        if (s.isUniform && input.isUniform) {
            return;
        }
        var texShapeA = s.texShape;
        var texShapeB = input.isUniform ? null : input.texData.texShape;
        if (!util.arraysEqual(texShapeA, texShapeB)) {
            throw Error("Binary was compiled with different texture shapes than the" +
                (" current args. Shape " + texShapeA + " and " + texShapeB + " must match"));
        }
    });
}
function runProgram(binary, inputs, output, customSetup) {
    validateBinaryAndProgram(binary.inShapeInfos, inputs);
    validateBinaryAndProgram([binary.outShapeInfo], [output]);
    var outTex = output.texData.texture;
    var outTexShape = output.texData.texShape;
    var gpgpu = binary.gpgpu;
    if (output.texData.isPacked) {
        gpgpu.setOutputPackedMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
    }
    else {
        gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
    }
    gpgpu.setProgram(binary.webGLProgram);
    inputs.forEach(function (input, i) {
        var variableName = binary.program.variableNames[i];
        var variableUniformLocation = binary.uniformLocations[variableName];
        if (variableUniformLocation != null) {
            if (input.isUniform) {
                if (util.sizeFromShape(input.shape) === 1) {
                    gpgpu.gl.uniform1f(variableUniformLocation, input.uniformValues[0]);
                }
                else {
                    var vals = input.uniformValues;
                    if (!(vals instanceof Float32Array)) {
                        vals = new Float32Array(vals);
                    }
                    gpgpu.gl.uniform1fv(variableUniformLocation, vals);
                }
                return;
            }
            var tex = input.texData.texture;
            gpgpu.setInputMatrixTexture(tex, variableUniformLocation, i);
        }
    });
    if (customSetup != null) {
        customSetup(gpgpu, binary.webGLProgram);
    }
    gpgpu.executeProgram();
}
exports.runProgram = runProgram;
function makeShaderKey(program, inputs, output) {
    var keyInputs = '';
    inputs.concat(output).forEach(function (x) {
        keyInputs += x.shape + "_" + (x.isUniform ? 'uniform' : x.texData.texShape);
    });
    var keyUserCode = program.userCode;
    var keyBroadcast = (program.supportsBroadcasting === true).toString();
    var key = program.constructor.name;
    key += '_' + keyBroadcast + '_' + keyInputs + '_' + keyUserCode;
    return key;
}
exports.makeShaderKey = makeShaderKey;
//# sourceMappingURL=gpgpu_math.js.map