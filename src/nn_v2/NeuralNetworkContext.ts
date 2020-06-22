import { OperandDescriptor } from './OperandDescriptor';
import { Input } from './Input';
import { Output } from './Output';
import { Model } from './Model';
import { Constant } from './Constant';
import { TypedArray } from './utils'
import { Operand } from './Operand';
import { Add } from './ops/Add';
import { Mul } from './ops/Mul';
import { OperandType } from './OperandType';

export class NeuralNetworkContext {
  constructor() {}

  async createModel(outputs: Array<Output>): Promise<Model> {
    return new Model(outputs);
  }

  input(desc: OperandDescriptor): Input {
    return new Input(desc);
  }

  constant(desc: OperandDescriptor, value: TypedArray): Constant;
  constant(value: number, type: OperandType): Constant;
  constant(descOrValue: any, valueOrType: any = 'float32'): Constant {
    return new Constant(descOrValue, valueOrType);
  }

  add(a: Operand, b: Operand): Operand {
    return (new Add(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    return (new Mul(a, b)).output;
  }
}