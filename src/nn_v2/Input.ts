import { OperandDescriptor } from './OperandDescriptor';
import { Operand } from './Operand';
import { assert, validateOperandDescriptor } from './utils';

export class Input extends Operand {
  readonly name: string;
  readonly desc: OperandDescriptor;

  constructor(name: string, desc: OperandDescriptor) {
    super();
    assert(typeof name === 'string', 'The name parameter is invalid');
    this.name = name;
    validateOperandDescriptor(desc);
    this.desc = desc;
  }
}