import { Tensor } from '../tensor';
import { Rank } from '../types';
import { DataType, ShapeMap } from '../types';
import { KernelBackend } from './backend';
export declare function castTensor<T extends Tensor>(x: T, dtype: DataType, backend: KernelBackend): T;
export declare function reshapeTensor<T extends Tensor, R extends Rank>(x: T, shape: ShapeMap[R]): Tensor<R>;
