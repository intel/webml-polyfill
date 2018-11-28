import { Tensor3D, Tensor4D } from '../tensor';
import { TensorLike } from '../types';
declare function localResponseNormalization_<T extends Tensor3D | Tensor4D>(x: T | TensorLike, depthRadius?: number, bias?: number, alpha?: number, beta?: number): T;
export declare const localResponseNormalization: typeof localResponseNormalization_;
export {};
