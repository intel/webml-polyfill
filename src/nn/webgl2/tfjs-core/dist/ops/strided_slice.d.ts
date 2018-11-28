import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function stridedSlice_<T extends Tensor>(x: T | TensorLike, begin: number[], end: number[], strides: number[], beginMask?: number, endMask?: number, ellipsisMask?: number, newAxisMask?: number, shrinkAxisMask?: number): T;
export declare const stridedSlice: typeof stridedSlice_;
export {};
