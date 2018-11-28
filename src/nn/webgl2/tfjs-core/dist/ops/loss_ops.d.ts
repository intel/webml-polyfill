import { Tensor } from '../tensor';
import { TensorLike } from '../types';
export declare enum Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2,
    SUM_BY_NONZERO_WEIGHTS = 3
}
declare function computeWeightedLoss_<T extends Tensor, O extends Tensor>(losses: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare function absoluteDifference_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare function meanSquaredError_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare function cosineDistance_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, axis: number, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare function hingeLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, reduction?: Reduction): O;
declare function logLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, epsilon?: number, reduction?: Reduction): O;
declare function sigmoidCrossEntropy_<T extends Tensor, O extends Tensor>(multiClassLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
declare function huberLoss_<T extends Tensor, O extends Tensor>(labels: T | TensorLike, predictions: T | TensorLike, weights?: Tensor | TensorLike, delta?: number, reduction?: Reduction): O;
declare function softmaxCrossEntropy_<T extends Tensor, O extends Tensor>(onehotLabels: T | TensorLike, logits: T | TensorLike, weights?: Tensor | TensorLike, labelSmoothing?: number, reduction?: Reduction): O;
export declare const absoluteDifference: typeof absoluteDifference_;
export declare const computeWeightedLoss: typeof computeWeightedLoss_;
export declare const cosineDistance: typeof cosineDistance_;
export declare const hingeLoss: typeof hingeLoss_;
export declare const huberLoss: typeof huberLoss_;
export declare const logLoss: typeof logLoss_;
export declare const meanSquaredError: typeof meanSquaredError_;
export declare const sigmoidCrossEntropy: typeof sigmoidCrossEntropy_;
export declare const softmaxCrossEntropy: typeof softmaxCrossEntropy_;
export {};
