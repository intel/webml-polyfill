import { Tensor } from '../tensor';
import { Rank, ShapeMap, TensorLike } from '../types';
declare function scatterND_<R extends Rank>(indices: Tensor | TensorLike, updates: Tensor | TensorLike, shape: ShapeMap[R]): Tensor<R>;
export declare const scatterND: typeof scatterND_;
export {};
