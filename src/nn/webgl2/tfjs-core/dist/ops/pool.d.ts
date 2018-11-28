import { Tensor3D, Tensor4D } from '../tensor';
import { TensorLike } from '../types';
declare function maxPool_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare function avgPool_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number, dimRoundingMode?: 'floor' | 'round' | 'ceil'): T;
declare function pool_<T extends Tensor3D | Tensor4D>(input: T | TensorLike, windowShape: [number, number] | number, poolingType: 'avg' | 'max', pad: 'valid' | 'same' | number, dilations?: [number, number] | number, strides?: [number, number] | number): T;
export declare const maxPool: typeof maxPool_;
export declare const avgPool: typeof avgPool_;
export declare const pool: typeof pool_;
export {};
