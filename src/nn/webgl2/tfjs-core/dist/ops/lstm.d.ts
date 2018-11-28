import { Scalar, Tensor1D, Tensor2D } from '../tensor';
import { TensorLike } from '../types';
export declare type LSTMCellFunc = {
    (data: Tensor2D, c: Tensor2D, h: Tensor2D): [Tensor2D, Tensor2D];
};
declare function multiRNNCell_(lstmCells: LSTMCellFunc[], data: Tensor2D | TensorLike, c: Tensor2D[] | TensorLike[], h: Tensor2D[] | TensorLike[]): [Tensor2D[], Tensor2D[]];
declare function basicLSTMCell_(forgetBias: Scalar | TensorLike, lstmKernel: Tensor2D | TensorLike, lstmBias: Tensor1D | TensorLike, data: Tensor2D | TensorLike, c: Tensor2D | TensorLike, h: Tensor2D | TensorLike): [Tensor2D, Tensor2D];
export declare const basicLSTMCell: typeof basicLSTMCell_;
export declare const multiRNNCell: typeof multiRNNCell_;
export {};
