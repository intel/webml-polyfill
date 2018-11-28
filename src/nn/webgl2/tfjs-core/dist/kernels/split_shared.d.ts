import { Tensor } from '../tensor';
export declare function split<T extends Tensor>(x: T, sizeSplits: number[], axis: number): T[];
