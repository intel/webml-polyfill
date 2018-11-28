import { GPGPUProgram } from './gpgpu_math';
export declare class UnpackProgram implements GPGPUProgram {
    variableNames: string[];
    usesPackedTextures: boolean;
    outputShape: number[];
    userCode: string;
    constructor(outputShape: number[]);
}
