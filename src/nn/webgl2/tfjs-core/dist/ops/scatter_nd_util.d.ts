import { Tensor } from '../tensor';
export declare function validateUpdateShape(shape: number[], indices: Tensor, updates: Tensor): void;
export interface ScatterShapeInfo {
    sliceRank: number;
    numUpdates: number;
    sliceSize: number;
    strides: number[];
    outputSize: number;
}
export declare function validateInput(updates: Tensor, indices: Tensor, shape: number[]): void;
export declare function calculateShapes(updates: Tensor, indices: Tensor, shape: number[]): ScatterShapeInfo;
