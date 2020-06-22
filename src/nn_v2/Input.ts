import { OperandDescriptor } from './OperandDescriptor';
import { Operand } from './Operand';
import { validateOperandDescriptor } from './utils';

export class Input extends Operand {
  readonly desc: OperandDescriptor;

  constructor(desc: OperandDescriptor) {
    super();
    validateOperandDescriptor(desc);
    this.desc = desc;
  }
}