import { Tensor } from '../tensor';
import { Rank, TensorLike } from '../types';
declare function gatherND_(x: Tensor | TensorLike, indices: Tensor | TensorLike): Tensor<Rank>;
export declare const gatherND: typeof gatherND_;
export {};
