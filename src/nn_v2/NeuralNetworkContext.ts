import { OperandDescriptor } from './OperandDescriptor';
import { Input } from './Input';
import { Output } from './Output';
import { Model } from './Model';
import { Constant } from './Constant';
import { TypedArray } from './utils'
import { Operand } from './Operand';
import { Add } from './ops/Add';
import { Mul } from './ops/Mul';

export class NeuralNetworkContext {
  constructor() {}

  async createModel(outputs: Array<Output>): Promise<Model> {
    return new Model(outputs);
  }

  input(desc: OperandDescriptor): Input {
    return new Input(desc);
  }

  constant(desc: OperandDescriptor, value: TypedArray): Constant {
    return new Constant(desc, value);
  }

  add(a: Operand, b: Operand): Operand {
    return (new Add(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    return (new Mul(a, b)).output;
  }
}