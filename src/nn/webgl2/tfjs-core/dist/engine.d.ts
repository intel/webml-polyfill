import { BackendTimingInfo, DataMover, KernelBackend } from './kernels/backend';
import { DataId, Tensor, Tensor3D, Variable } from './tensor';
import { NamedTensorMap, NamedVariableMap, TensorContainer } from './tensor_types';
import { TypedArray } from './types';
export declare type ForwardFunc<T> = (backend: KernelBackend, save?: <S extends Tensor>(tensor: S) => S) => T;
export declare type CustomGradientFunc<T extends Tensor> = (...args: Tensor[]) => {
    value: T;
    gradFunc: (dy: T) => Tensor | Tensor[];
};
export declare type MemoryInfo = {
    numTensors: number;
    numDataBuffers: number;
    numBytes: number;
    unreliable?: boolean;
};
declare type KernelProfile = {
    name: string;
    bytesAdded: number;
    totalBytesSnapshot: number;
    tensorsAdded: number;
    totalTensorsSnapshot: number;
    inputShapes: number[][];
    outputShape: number[] | number[][];
};
export declare type ProfileInfo = {
    newBytes: number;
    newTensors: number;
    peakBytes: number;
    kernels: KernelProfile[];
    result: TensorContainer;
};
export interface TimingInfo extends BackendTimingInfo {
    wallMs: number;
}
export declare type ScopeFn<T extends TensorContainer> = () => T;
export interface TensorManager {
    registerTensor(a: Tensor): void;
    registerVariable(v: Variable): void;
    disposeTensor(a: Tensor): void;
    memory(): {
        numDataBuffers: number;
        numBytes: number;
    };
}
export declare class Engine implements TensorManager, DataMover {
    backend: KernelBackend;
    safeMode: boolean;
    private debugMode;
    registeredVariables: NamedVariableMap;
    private nextTapeNodeId;
    private numBytes;
    private numTensors;
    private numDataBuffers;
    private profiling;
    private activeProfile;
    private activeTape;
    private gradientScopeCount;
    private customGradientDepth;
    private activeScope;
    private scopeStack;
    private keepTensors;
    private profiler;
    private tensorInfo;
    constructor(backend: KernelBackend, safeMode: boolean, debugMode: () => boolean);
    moveData(dataId: DataId): void;
    tidy<T extends TensorContainer>(nameOrFn: string | ScopeFn<T>, fn?: ScopeFn<T>, gradMode?: boolean): T;
    private scopedRun;
    private static nextTensorId;
    nextTensorId(): number;
    private static nextVariableId;
    nextVariableId(): number;
    runKernel<T extends Tensor | Tensor[], I extends NamedTensorMap>(forwardFunc: ForwardFunc<T>, inputs: I, backwardsFunc?: (dy: T, saved: Tensor[]) => {
        [P in keyof I]: () => I[P];
    }): T;
    registerTensor(a: Tensor | Variable): void;
    registerVariable(v: Variable): void;
    disposeTensor(a: Tensor): void;
    disposeVariables(): void;
    memory(): MemoryInfo;
    profile(query: () => TensorContainer): Promise<ProfileInfo>;
    private shouldRecord;
    private addTapeNode;
    keep<T extends Tensor>(result: T): T;
    startScope(name?: string, gradientsMode?: boolean): void;
    endScope(result?: TensorContainer, gradientsMode?: boolean): void;
    gradients<T extends Tensor>(f: () => T, xs: Tensor[], dy?: T, allowNoGradients?: boolean): {
        value: T;
        grads: Tensor[];
    };
    customGrad<T extends Tensor>(f: CustomGradientFunc<T>): (...args: Tensor[]) => T;
    write(dataId: DataId, values: TypedArray): void;
    readSync(dataId: DataId): TypedArray;
    read(dataId: DataId): Promise<TypedArray>;
    fromPixels(pixels: ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, numChannels: number): Tensor3D;
    time(query: () => void): Promise<TimingInfo>;
    private track;
}
export {};
