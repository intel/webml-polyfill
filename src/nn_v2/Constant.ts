import { Input } from './Input'
import { OperandDescriptor } from './OperandDescriptor';
import { assert, TypedArray, isTypedArray, getTypedArray, isTensorType, getDataType } from './utils';
import * as tf from '@tensorflow/tfjs-core'

export class Constant extends Input {
  readonly tensor: tf.Tensor;

  constructor(desc: OperandDescriptor, value: TypedArray) {
    super(desc);
    this.validateValue(value);
    this.tensor = this.createTensor(value);
  }

  private validateValue(value: TypedArray) {
    assert(isTypedArray(value), 'The value is not a typed array.');
    assert(value instanceof getTypedArray(this.desc.type), 'The type of value is invalid.');
    if (!isTensorType(this.desc.type)) {
      assert(value.length === 1, 'The value length is invalid.');
    }
  }

  private createTensor(value: TypedArray): tf.Tensor {
    const dtype: tf.DataType = getDataType(this.desc.type);
    if (isTensorType(this.desc.type)) {
      return tf.tensor(value, this.desc.dimensions, dtype);
    } else {
      return tf.scalar(value[0], dtype);
    }
  }
}