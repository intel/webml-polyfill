import { BackendTimer } from './kernels/backend';
import { Tensor } from './tensor';
import { TypedArray } from './types';
export declare class Profiler {
    private backendTimer;
    private logger?;
    constructor(backendTimer: BackendTimer, logger?: Logger);
    profileKernel<T extends Tensor | Tensor[]>(name: string, f: () => T | Tensor[]): T;
}
export declare class Logger {
    logKernelProfile(name: string, result: Tensor, vals: TypedArray, timeMs: number, extraInfo?: string): void;
}
