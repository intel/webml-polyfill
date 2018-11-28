"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var broadcast_util = require("../../ops/broadcast_util");
var BatchNormPackedProgram = (function () {
    function BatchNormPackedProgram(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
        this.supportsBroadcasting = true;
        this.usesPackedTextures = true;
        this.variableNames = ['x', 'mean', 'variance'];
        broadcast_util.assertAndGetBroadcastShape(xShape, meanShape);
        broadcast_util.assertAndGetBroadcastShape(xShape, varianceShape);
        var meanSnippet = broadcastSample('mean', meanShape.length);
        var varianceSnippet = broadcastSample('variance', varianceShape.length);
        var offsetSnippet = 'vec4 offset = vec4(0.0)';
        if (offsetShape != null) {
            broadcast_util.assertAndGetBroadcastShape(xShape, offsetShape);
            this.variableNames.push('offset');
            offsetSnippet = broadcastSample('offset', offsetShape.length);
        }
        var scaleSnippet = 'vec4 scale = vec4(1.0)';
        if (scaleShape != null) {
            broadcast_util.assertAndGetBroadcastShape(xShape, scaleShape);
            this.variableNames.push('scale');
            scaleSnippet = broadcastSample('scale', scaleShape.length);
        }
        this.outputShape = xShape;
        this.userCode = "\n      void main() {\n        ivec4 rc = getOutputCoords();\n\n        " + offsetSnippet + ";\n        " + scaleSnippet + ";\n\n        vec4 x = getX(rc.x, rc.y, rc.z, rc.w);\n        " + meanSnippet + ";\n        " + varianceSnippet + ";\n\n        vec4 inv = scale * inversesqrt(variance + vec4(" + varianceEpsilon + "));\n\n        setOutput((x - mean) * inv + offset);\n      }\n    ";
    }
    return BatchNormPackedProgram;
}());
exports.BatchNormPackedProgram = BatchNormPackedProgram;
function broadcastSample(texName, rank) {
    var texSampler = "get" + texName.charAt(0).toUpperCase() + texName.slice(1);
    if (rank === 1) {
        return "\n      vec4 " + texName + "Sample = " + texSampler + "(rc.w);\n      vec4 " + texName + " = vec4(" + texName + "Sample.xy, " + texName + "Sample.xy);\n    ";
    }
    return "vec4 " + texName + " = " + texSampler + "(rc.x, rc.y, rc.z, rc.w)";
}
//# sourceMappingURL=batchnorm_packed_gpu.js.map