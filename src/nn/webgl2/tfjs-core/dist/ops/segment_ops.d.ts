import { Tensor, Tensor1D } from '../tensor';
import { TensorLike } from '../types';
declare function unsortedSegmentSum_<T extends Tensor>(x: T | TensorLike, segmentIds: Tensor1D | TensorLike, numSegments: number): T;
declare function gather_<T extends Tensor>(x: T | TensorLike, indices: Tensor1D | TensorLike, axis?: number): T;
export declare const gather: typeof gather_;
export declare const unsortedSegmentSum: typeof unsortedSegmentSum_;
export {};
