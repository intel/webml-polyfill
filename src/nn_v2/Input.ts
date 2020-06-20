import { OperandDescriptor } from './OperandDescriptor';
import { Operand } from './Operand'
import { OperandType } from './OperandType'
import { assert, isTensorType, isNumberArray } from './utils';

export class Input extends Operand {
  protected desc: OperandDescriptor;

  constructor(desc: OperandDescriptor) {
    super();
    this.validateOperandDescriptor(desc);
    this.desc = desc;
  }

  private validateOperandDescriptor(desc: OperandDescriptor) {
    assert(desc.type in OperandType, 'The operand type is invalid.');
    if (isTensorType(desc.type)) {
      assert(isNumberArray(desc.dimensions), 'The operand dimensions is invalid.');
    } else {
      assert(desc.dimensions === undefined, 'The operand dimensions is not required.');
    }
  }
}