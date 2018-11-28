import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function softmax_<T extends Tensor>(logits: T | TensorLike, dim?: number): T;
declare function logSoftmax_<T extends Tensor>(logits: T | TensorLike, axis?: number): T;
export declare const softmax: typeof softmax_;
export declare const logSoftmax: typeof logSoftmax_;
export {};
