import { GPGPUProgram } from './gpgpu_math';
export declare class ScatterProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(updateSize: number, sliceDim: number, indicesRank: number, updatesRank: number, strides: number[], shape: number[], summingDupeIndex?: boolean);
}
