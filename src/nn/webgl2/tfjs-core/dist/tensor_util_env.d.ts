import { Tensor } from './tensor';
import { DataType, RegularArray, TensorLike, TypedArray } from './types';
export declare function inferShape(val: TypedArray | number | boolean | RegularArray<number> | RegularArray<boolean>): number[];
export declare function convertToTensor<T extends Tensor>(x: T | TensorLike, argName: string, functionName: string, dtype?: DataType): T;
export declare function convertToTensorArray<T extends Tensor>(arg: T[] | TensorLike[], argName: string, functionName: string): T[];
