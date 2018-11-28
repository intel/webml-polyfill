import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function topk_<T extends Tensor>(x: T | TensorLike, k?: number, sorted?: boolean): {
    values: T;
    indices: T;
};
export declare const topk: typeof topk_;
export {};
