import { GPGPUProgram } from './gpgpu_math';
export declare class LRNGradProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    depthRadius: number;
    bias: number;
    alpha: number;
    beta: number;
    depth: number;
    constructor(inputShape: number[], depthRadius: number, bias: number, alpha: number, beta: number);
}
