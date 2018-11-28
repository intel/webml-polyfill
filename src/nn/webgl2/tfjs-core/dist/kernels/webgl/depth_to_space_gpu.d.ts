import { GPGPUProgram } from './gpgpu_math';
export declare class DepthToSpaceProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    blockSize: number;
    dataFormat: string;
    constructor(outputShape: number[], blockSize: number, dataFormat: 'NHWC' | 'NCHW');
    private getHeightCoordString;
    private getWidthCoordString;
    private getDepthCoordString;
    private getOutputDepthSize;
    private getInputSamplingString;
}
