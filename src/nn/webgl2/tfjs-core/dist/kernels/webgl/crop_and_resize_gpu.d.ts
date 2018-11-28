import { GPGPUProgram } from './gpgpu_math';
export declare class CropAndResizeProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(imageShape: [number, number, number, number], boxShape: [number, number], cropSize: [number, number], method: 'bilinear' | 'nearest', extrapolationValue: number);
}
