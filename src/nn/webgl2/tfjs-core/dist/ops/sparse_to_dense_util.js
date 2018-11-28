"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function validateInput(sparseIndices, sparseValues, outputShape, defaultValues) {
    if (sparseIndices.dtype !== 'int32') {
        throw new Error('tf.sparseToDense() expects the indices to be int32 type,' +
            (" but the dtype was " + sparseIndices.dtype + "."));
    }
    if (sparseIndices.rank > 2) {
        throw new Error('sparseIndices should be a scalar, vector, or matrix,' +
            (" but got shape " + sparseIndices.shape + "."));
    }
    var numElems = sparseIndices.rank > 0 ? sparseIndices.shape[0] : 1;
    var numDims = sparseIndices.rank > 1 ? sparseIndices.shape[1] : 1;
    if (outputShape.length !== numDims) {
        throw new Error('outputShape has incorrect number of elements:,' +
            (" " + outputShape.length + ", should be: " + numDims + "."));
    }
    var numValues = sparseValues.size;
    if (!(sparseValues.rank === 0 ||
        sparseValues.rank === 1 && numValues === numElems)) {
        throw new Error('sparseValues has incorrect shape ' +
            (sparseValues.shape + ", should be [] or [" + numElems + "]"));
    }
    if (sparseValues.dtype !== defaultValues.dtype) {
        throw new Error('sparseValues.dtype must match defaultValues.dtype');
    }
}
exports.validateInput = validateInput;
//# sourceMappingURL=sparse_to_dense_util.js.map