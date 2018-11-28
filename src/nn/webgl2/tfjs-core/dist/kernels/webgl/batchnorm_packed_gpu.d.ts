import { GPGPUProgram } from './gpgpu_math';
export declare class BatchNormPackedProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    supportsBroadcasting: boolean;
    usesPackedTextures: boolean;
    constructor(xShape: number[], meanShape: number[], varianceShape: number[], offsetShape: number[] | null, scaleShape: number[] | null, varianceEpsilon: number);
}
