import { Scalar, Tensor } from '../tensor';
import { Rank, ShapeMap, TensorLike } from '../types';
declare function sparseToDense_<R extends Rank>(sparseIndices: Tensor | TensorLike, sparseValues: Tensor | TensorLike, outputShape: ShapeMap[R], defaultValue: Scalar | TensorLike): Tensor<R>;
export declare const sparseToDense: typeof sparseToDense_;
export {};
