import { Tensor } from '../tensor';
import { DataType, TypedArray } from '../types';
export declare function topkImpl<T extends Tensor>(x: TypedArray, xShape: number[], xDtype: DataType, k: number, sorted: boolean): [T, T];
