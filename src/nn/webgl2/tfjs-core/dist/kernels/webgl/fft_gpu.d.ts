import { GPGPUProgram } from './gpgpu_math';
export declare const COMPLEX_FFT: {
    REAL: string;
    IMAG: string;
};
export declare class FFTProgram implements GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    constructor(op: string, inputShape: [number, number], inverse: boolean);
}
