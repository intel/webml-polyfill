import { GPGPUProgram } from './gpgpu_math';
export declare class SelectProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(cRank: number, shape: number[], rank: number);
}
