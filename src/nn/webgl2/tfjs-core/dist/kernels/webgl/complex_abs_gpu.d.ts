import { GPGPUProgram } from './gpgpu_math';
export declare class ComplexAbsProgram implements GPGPUProgram {
    variableNames: string[];
    userCode: string;
    outputShape: number[];
    constructor(shape: number[]);
}
