import { Tensor, Variable } from './tensor';
import { DataType } from './types';
export declare type NamedTensorMap = {
    [name: string]: Tensor;
};
export declare type NamedVariableMap = {
    [name: string]: Variable;
};
export declare type TensorContainer = void | Tensor | string | number | boolean | TensorContainerObject | TensorContainerArray;
export interface TensorContainerObject {
    [x: string]: TensorContainer;
}
export interface TensorContainerArray extends Array<TensorContainer> {
}
export interface TensorInfo {
    name: string;
    shape?: number[];
    dtype: DataType;
}
