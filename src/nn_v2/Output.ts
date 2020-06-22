import { Operand } from "./Operand";
import { Operation } from "./Operation";

export class Output extends Operand {
  readonly operation: Operation;

  constructor(operation: Operation) {
    super();
    this.operation = operation;
  }
}