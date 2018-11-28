import { Conv2DInfo } from '../../ops/conv_util';
import { GPGPUProgram } from './gpgpu_math';
export declare class Im2ColProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(outputShape: number[], inputShape: number[], convInfo: Conv2DInfo);
}
