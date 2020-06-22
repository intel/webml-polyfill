import { Output } from './Output';
import { Input } from './Input';
import { Operation } from './Operation';
import { CompilationOptions } from './CompilationOptions';
import { Compilation } from './Compilation';
import { assert } from './utils';

export class Model {
  inputs_: Array<Input> = [];
  outputs_: Array<Output> = [];

  constructor(outputs: Array<Output>) {
    assert(outputs.length !== 0, 'The length of outputs parameter should not be 0.');
    assert(outputs.every(output => output instanceof Output), 'The outputs parameter is invalid.');
    this.outputs_ = outputs;
    this.identifyInputs();
  }

  async createCompilation(options: CompilationOptions): Promise<Compilation> {
    // TODO: compile the model, e.g. run once.
    return new Compilation(options, this);
  }

  private identifyInputs(): void {
    const self = this;
    function handleOperation(operation: Operation): void {
      for (const operand of operation.inputs) {
        if (operand instanceof Input) {
          self.inputs_.push((operand as Input));
        } else if (operand instanceof Output) {
          handleOperation((operand as Output).operation);
        }
      }
    }
    for (const output of this.outputs_) {
      handleOperation(output.operation);
    }
  }
}