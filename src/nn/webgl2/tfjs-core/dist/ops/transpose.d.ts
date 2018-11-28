import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function transpose_<T extends Tensor>(x: T | TensorLike, perm?: number[]): T;
export declare const transpose: typeof transpose_;
export {};
