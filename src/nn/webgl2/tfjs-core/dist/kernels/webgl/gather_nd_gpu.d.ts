import { GPGPUProgram } from './gpgpu_math';
export declare class GatherNDProgram implements GPGPUProgram {
    private sliceDim;
    private strides;
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(sliceDim: number, strides: number[], shape: number[]);
}
