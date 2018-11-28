import { GPGPUProgram } from './gpgpu_math';
export declare class MatMulPackedProgram implements GPGPUProgram {
    variableNames: string[];
    usesPackedTextures: boolean;
    outputShape: number[];
    userCode: string;
    constructor(aShape: [number, number], bShape: [number, number], outputShape: [number, number], transposeA?: boolean, transposeB?: boolean);
}
