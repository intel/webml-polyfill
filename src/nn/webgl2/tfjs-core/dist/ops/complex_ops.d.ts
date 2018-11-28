import { Tensor } from '../tensor';
import { TensorLike } from '../types';
declare function complex_<T extends Tensor>(real: T | TensorLike, imag: T | TensorLike): T;
declare function real_<T extends Tensor>(input: T | TensorLike): T;
declare function imag_<T extends Tensor>(input: T | TensorLike): T;
export declare const complex: typeof complex_;
export declare const real: typeof real_;
export declare const imag: typeof imag_;
export {};
