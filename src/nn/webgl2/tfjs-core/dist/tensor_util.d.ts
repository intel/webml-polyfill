import { Tensor } from './tensor';
import { NamedTensorMap, TensorContainer } from './tensor_types';
export declare function assertTypesMatch(a: Tensor, b: Tensor): void;
export declare function isTensorInList(tensor: Tensor, tensorList: Tensor[]): boolean;
export declare function flattenNameArrayMap(nameArrayMap: Tensor | NamedTensorMap, keys?: string[]): Tensor[];
export declare function unflattenToNameArrayMap(keys: string[], flatArrays: Tensor[]): NamedTensorMap;
export declare function getTensorsInContainer(result: TensorContainer): Tensor[];
