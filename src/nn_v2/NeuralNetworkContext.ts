import { OperandDescriptor } from './OperandDescriptor';
import { Input } from './Input';
import { Constant } from './Constant';
import { TypedArray } from './utils'

export class NeuralNetworkContext {
  constructor() {}

  input(desc: OperandDescriptor): Input {
    return new Input(desc);
  }

  constant(desc: OperandDescriptor, value: TypedArray): Constant {
    return new Constant(desc, value);
  }
}