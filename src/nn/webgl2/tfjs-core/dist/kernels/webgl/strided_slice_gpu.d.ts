import { GPGPUProgram } from './gpgpu_math';
export declare class StridedSliceProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(begin: number[], strides: number[], size: number[], shrinkAxis: number[]);
}
