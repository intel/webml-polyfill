import { GPGPUProgram } from './gpgpu_math';
export declare class PackProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(outputShape: number[]);
}
