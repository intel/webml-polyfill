import { Scalar, Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function movingAverage_<T extends Tensor>(v: T | TensorLike, x: T | TensorLike, decay: number | Scalar, step?: number | Scalar, zeroDebias?: boolean): T;
export declare const movingAverage: typeof movingAverage_;
export {};
