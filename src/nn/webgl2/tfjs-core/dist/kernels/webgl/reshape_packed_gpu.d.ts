import { GPGPUProgram } from './gpgpu_math';
export declare class ReshapePackedProgram implements GPGPUProgram {
    variableNames: string[];
    usesPackedTextures: boolean;
    outputShape: number[];
    userCode: string;
    constructor(outputShape: [number, number, number], inputShape: [number, number, number]);
}
