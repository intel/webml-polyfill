import { Tensor, Tensor1D, Tensor2D } from '../tensor';
import { TensorLike } from '../types';
declare function matMul_<T extends Tensor>(a: T | TensorLike, b: T | TensorLike, transposeA?: boolean, transposeB?: boolean): T;
declare function outerProduct_(v1: Tensor1D | TensorLike, v2: Tensor1D | TensorLike): Tensor2D;
declare function dot_(t1: Tensor | TensorLike, t2: Tensor | TensorLike): Tensor;
export declare const matMul: typeof matMul_;
export declare const dot: typeof dot_;
export declare const outerProduct: typeof outerProduct_;
export {};
