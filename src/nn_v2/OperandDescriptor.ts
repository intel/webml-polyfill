import { OperandType } from './OperandType'

export interface OperandDescriptor {
  type: OperandType;
  dimensions: Array<number>;
}