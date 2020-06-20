import { OperandType } from './OperandType'
import * as tf from '@tensorflow/tfjs-core'

export type TypedArray = Float32Array|Int32Array|Uint32Array|Int16Array|Uint16Array;

export function assert(expr: boolean, msg: string) {
  if (!expr) {
    throw new Error(msg);
  }
}

export function isNumber(value: {}): boolean {
  return typeof value === 'number';
}

export function isNumberArray(array: number[]): boolean {
  return array instanceof Array && array.every(v => isNumber(v));
}

export function isTensorType(type: OperandType): boolean {
  return type === 'tensor-float32' || type === 'tensor-float16' ||
      type === 'tensor-int32';
}

export function isTypedArray(array: TypedArray): boolean {
  return array instanceof Float32Array || array instanceof Int32Array ||
      array instanceof Uint32Array || array instanceof Int16Array ||
      array instanceof Uint16Array;
}

export function getTypedArray(type: OperandType) {
  if (type === 'float32' || type === 'tensor-float32') {
    return Float32Array;
  } else if (type === 'int32' || type === 'tensor-int32') {
    return Int32Array;
  } else if (type === 'uint32') {
    return Uint32Array;
  } else if (type === 'float16' || type === 'tensor-float16') {
    return Uint16Array;
  } else {
    throw new Error('Type is not supported.');
  }
}

export function getDataType(type: OperandType): tf.DataType {
  if (type === 'float32' || type === 'tensor-float32') {
    return 'float32';
  } else if (type === 'int32' || type === 'tensor-int32') {
    return 'int32';
  } else {
    throw new Error('The operand type is not supported by TF.js.');
  }
}