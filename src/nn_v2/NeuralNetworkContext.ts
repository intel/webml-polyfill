import { OperandDescriptor } from './OperandDescriptor';
import { Input } from './Input';
import { Model } from './Model';
import { Constant } from './Constant';
import { TypedArray } from './utils'
import { Operand } from './Operand';
import { Add } from './ops/Add';
import { Mul } from './ops/Mul';
import { OperandType } from './OperandType';
import { NamedOperand } from './NamedOperand';

export class NeuralNetworkContext {
  constructor() {}

  async createModel(outputs: Array<NamedOperand>): Promise<Model> {
    return new Model(outputs);
  }

  input(name: string, desc: OperandDescriptor): Input {
    return new Input(name, desc);
  }

  constant(desc: OperandDescriptor, value: TypedArray): Constant;
  constant(value: number, type: OperandType): Constant;
  constant(descOrValue: OperandDescriptor|number, valueOrType: TypedArray|OperandType): Constant {
    if (typeof descOrValue === 'number') {
      return Constant.createScalar(descOrValue as number, valueOrType as OperandType);
    } else {
      return Constant.createTensor(descOrValue as OperandDescriptor, valueOrType as TypedArray);
    }
  }

  add(a: Operand, b: Operand): Operand {
    return (new Add(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    return (new Mul(a, b)).output;
  }
}