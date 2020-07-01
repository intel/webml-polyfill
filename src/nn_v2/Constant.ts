import { Operand } from './Operand'
import { OperandDescriptor } from './OperandDescriptor';
import { assert, TypedArray, validateTypedArray, createTensor, validateOperandDescriptor, getDataType } from './utils';
import { OperandType } from './OperandType';

import * as tf from '@tensorflow/tfjs-core'

export class Constant extends Operand {
  private desc_: OperandDescriptor;
  private value_: number|TypedArray;

  get desc() { return this.desc_; }

  createTensor(): tf.Tensor {
    if (typeof this.value_ === 'number') {
      const dtype: tf.DataType = getDataType(this.desc_.type);
      return tf.scalar(this.value_, dtype);
    } else {
      return createTensor(this.desc_, this.value_);
    }
  }

  static createScalar(value: number, type: OperandType = OperandType.float32): Constant {
    let constant = new Constant();
    if (typeof type === 'undefined') {
      type = OperandType.float32;
    }
    assert(type in OperandType, 'The operand type is invalid.');
    constant.desc_ = {type: type} as OperandDescriptor;
    constant.value_ = value;
    return constant;
  }

  static createTensor(desc: OperandDescriptor, value: TypedArray): Constant {
    let constant = new Constant();
    validateOperandDescriptor(desc);
    constant.desc_ = desc;
    validateTypedArray(value, desc);
    constant.value_ = value;
    return constant;
  }

  private constructor() {
    super();
  }
}