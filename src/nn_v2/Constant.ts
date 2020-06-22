import { Operand } from './Operand'
import { OperandDescriptor } from './OperandDescriptor';
import { TypedArray, validateTypedArray, createTensor, validateOperandDescriptor } from './utils';
import * as tf from '@tensorflow/tfjs-core'

export class Constant extends Operand {
  readonly desc: OperandDescriptor;
  readonly tensor: tf.Tensor;

  constructor(desc: OperandDescriptor, value: TypedArray) {
    super();
    validateOperandDescriptor(desc);
    this.desc = desc;
    validateTypedArray(value, this.desc.type);
    this.tensor = createTensor(this.desc, value);
  }
}