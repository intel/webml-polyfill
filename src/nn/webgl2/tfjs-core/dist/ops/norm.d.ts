import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function norm_(x: Tensor | TensorLike, ord?: number | 'euclidean' | 'fro', axis?: number | number[], keepDims?: boolean): Tensor;
export declare const norm: typeof norm_;
export {};
