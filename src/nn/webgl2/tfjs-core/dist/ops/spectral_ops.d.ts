import { Tensor } from '../tensor';
declare function fft_(input: Tensor): Tensor;
declare function ifft_(input: Tensor): Tensor;
declare function rfft_(input: Tensor): Tensor;
export declare const fft: typeof fft_;
export declare const ifft: typeof ifft_;
export declare const rfft: typeof rfft_;
export {};
