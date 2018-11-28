import { Tensor1D, Tensor2D } from '../tensor';
import { TensorLike } from '../types';
export declare function confusionMatrix_(labels: Tensor1D | TensorLike, predictions: Tensor1D | TensorLike, numClasses: number): Tensor2D;
export declare const confusionMatrix: typeof confusionMatrix_;
