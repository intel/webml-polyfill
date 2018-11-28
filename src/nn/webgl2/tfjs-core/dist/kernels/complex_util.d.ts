import { TypedArray } from '../types';
export declare function mergeRealAndImagArrays(real: Float32Array, imag: Float32Array): Float32Array;
export declare function splitRealAndImagArrays(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
export declare function complexWithEvenIndex(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
export declare function complexWithOddIndex(complex: Float32Array): {
    real: Float32Array;
    imag: Float32Array;
};
export declare function getComplexWithIndex(complex: Float32Array, index: number): {
    real: number;
    imag: number;
};
export declare function assignToTypedArray(data: TypedArray, real: number, imag: number, index: number): void;
export declare function exponents(n: number, inverse: boolean): {
    real: Float32Array;
    imag: Float32Array;
};
export declare function exponent(k: number, n: number, inverse: boolean): {
    real: number;
    imag: number;
};
