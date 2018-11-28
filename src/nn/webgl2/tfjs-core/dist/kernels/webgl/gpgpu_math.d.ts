import { Tensor } from '../../tensor';
import { TypedArray } from '../../types';
import { GPGPUContext } from './gpgpu_context';
import { ShapeInfo } from './shader_compiler';
import { TextureData } from './tex_util';
export interface GPGPUProgram {
    variableNames: string[];
    outputShape: number[];
    userCode: string;
    usesPackedTextures?: boolean;
    supportsBroadcasting?: boolean;
}
export interface GPGPUBinary {
    webGLProgram: WebGLProgram;
    program: GPGPUProgram;
    uniformLocations: {
        [name: string]: WebGLUniformLocation;
    };
    gpgpu: GPGPUContext;
    source: string;
    inShapeInfos: ShapeInfo[];
    outShapeInfo: ShapeInfo;
}
export interface TensorData {
    shape: number[];
    texData: TextureData;
    isUniform: boolean;
    uniformValues?: TypedArray;
}
export declare function compileProgram<T extends Tensor, K extends Tensor>(gpgpu: GPGPUContext, program: GPGPUProgram, inputs: TensorData[], output: TensorData): GPGPUBinary;
export declare function runProgram<T extends Tensor, K extends Tensor>(binary: GPGPUBinary, inputs: TensorData[], output: TensorData, customSetup?: (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => void): void;
export declare function makeShaderKey(program: GPGPUProgram, inputs: TensorData[], output: TensorData): string;
