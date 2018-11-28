import { GPGPUProgram } from './gpgpu_math';
export declare class MatMulProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(aShape: [number, number, number], bShape: [number, number, number], transposeA?: boolean, transposeB?: boolean);
}
