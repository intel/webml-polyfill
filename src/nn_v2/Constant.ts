import { Operand } from './Operand'
import { OperandDescriptor } from './OperandDescriptor';
import { assert, TypedArray, validateTypedArray, createTensor, validateOperandDescriptor, getDataType } from './utils';
import * as tf from '@tensorflow/tfjs-core'
import { OperandType } from './OperandType';

export class Constant extends Operand {
  private desc_: OperandDescriptor;
  private tensor_: tf.Tensor;

  get desc() { return this.desc_; }
  get tensor() { return this.tensor_; }

  static createScalar(value: number, type: OperandType = OperandType.float32): Constant {
    let constant = new Constant();
    if (typeof type === 'undefined') {
      type = OperandType.float32;
    }
    assert(type in OperandType, 'The operand type is invalid.');
    constant.desc_ = {type: type} as OperandDescriptor;
    const dtype: tf.DataType = getDataType(type);
    constant.tensor_ = tf.scalar(value, dtype);
    return constant;
  }

  static createTensor(desc: OperandDescriptor, value: TypedArray): Constant {
    let constant = new Constant();
    validateOperandDescriptor(desc);
    constant.desc_ = desc;
    validateTypedArray(value, desc.type);
    constant.tensor_ = createTensor(desc, value);
    return constant;
  }

  private constructor() {
    super();
  }
}