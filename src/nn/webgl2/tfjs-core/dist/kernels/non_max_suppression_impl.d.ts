import { Tensor1D } from '../tensor';
import { TypedArray } from '../types';
export declare function nonMaxSuppressionImpl(boxes: TypedArray, scores: TypedArray, maxOutputSize: number, iouThreshold: number, scoreThreshold: number): Tensor1D;
