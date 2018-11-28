import { Tensor } from '../../tensor';
import { DataType, DataTypeMap } from '../../types';
export declare enum TextureUsage {
    RENDER = 0,
    UPLOAD = 1,
    PIXELS = 2,
    DOWNLOAD = 3
}
export declare enum PhysicalTextureType {
    UNPACKED_FLOAT16 = 0,
    UNPACKED_FLOAT32 = 1,
    PACKED_4X1_UNSIGNED_BYTE = 2,
    PACKED_2X2_FLOAT32 = 3,
    PACKED_2X2_FLOAT16 = 4
}
export interface TextureData {
    texture: WebGLTexture;
    complexTensors?: {
        real: Tensor;
        imag: Tensor;
    };
    shape: number[];
    texShape: [number, number];
    dtype: DataType;
    values: DataTypeMap[DataType];
    usage: TextureUsage;
    isPacked: boolean;
}
export declare function getUnpackedMatrixTextureShapeWidthHeight(rows: number, columns: number): [number, number];
export declare function getUnpackedArraySizeFromMatrixSize(matrixSize: number, channelsPerTexture: number): number;
export declare function getColorMatrixTextureShapeWidthHeight(rows: number, columns: number): [number, number];
export declare function getMatrixSizeFromUnpackedArraySize(unpackedSize: number, channelsPerTexture: number): number;
export declare type TypedArray = Float32Array | Uint8Array;
export declare function encodeMatrixToUnpackedArray(matrix: TypedArray, unpackedArray: TypedArray, channelsPerTexture: number): void;
export declare function decodeMatrixFromUnpackedArray(unpackedArray: Float32Array, matrix: Float32Array, channelsPerTexture: number): void;
export declare function decodeMatrixFromUnpackedColorRGBAArray(unpackedArray: Float32Array, matrix: Float32Array, channels: number): void;
export declare function getPackedMatrixTextureShapeWidthHeight(rows: number, columns: number): [number, number];
export declare function getPackedRGBAArraySizeFromMatrixShape(rows: number, columns: number): number;
export declare function encodeMatrixToPackedRGBA(matrix: Float32Array, batches: number, rows: number, columns: number, packedRGBA: Float32Array): Float32Array;
export declare function decodeMatrixFromPackedRGBA(packedRGBA: Float32Array, batches: number, rows: number, columns: number, matrix: Float32Array): Float32Array;
