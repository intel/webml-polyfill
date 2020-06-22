import { Operand } from './Operand'
import { OperandDescriptor } from './OperandDescriptor';
import { assert, TypedArray, validateTypedArray, createTensor, validateOperandDescriptor, getDataType } from './utils';
import * as tf from '@tensorflow/tfjs-core'
import { OperandType } from './OperandType';

export class Constant extends Operand {
  readonly desc: OperandDescriptor;
  readonly tensor: tf.Tensor;

  constructor(desc: OperandDescriptor, value: TypedArray);
  constructor(value: number, type: OperandType);
  constructor(descOrValue: any, valueOrType: any) {
    super();
    if (typeof descOrValue === 'number') {
      const type = valueOrType as OperandType;
      const value = descOrValue as number;
      assert(type in OperandType, 'The operand type is invalid.');
      this.desc = {type: type} as OperandDescriptor;
      const dtype: tf.DataType = getDataType(type);
      this.tensor = tf.scalar(value, dtype);
    } else {
      const desc = descOrValue as OperandDescriptor;
      const value = valueOrType as TypedArray;
      validateOperandDescriptor(desc);
      this.desc = desc;
      validateTypedArray(value, this.desc.type);
      this.tensor = createTensor(this.desc, value);
    }
  }
}