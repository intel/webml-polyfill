import { Output } from './Output';
import { Input } from './Input';
import { Operation } from './Operation';
import { CompilationOptions } from './CompilationOptions';
import { Compilation } from './Compilation';
import { assert } from './utils';

export class Model {
  inputs: Array<Input> = [];
  outputs: Array<Output> = [];

  constructor(outputs: Array<Output>) {
    assert(outputs.length !== 0, 'The length of outputs parameter should not be 0.');
    assert(outputs.every(output => output instanceof Output), 'The outputs parameter is invalid.');
    this.outputs = outputs;
    this.identifyInputs();
  }

  async createCompilation(options: CompilationOptions): Promise<Compilation> {
    return new Compilation(options, this);
  }

  private identifyInputs(): void {
    const self = this;
    function handleOperation(operation: Operation): void {
      for (const operand of operation.inputs) {
        if (operand instanceof Input) {
          self.inputs.push((operand as Input));
        } else if (operand instanceof Output) {
          handleOperation((operand as Output).operation);
        }
      }
    }
    for (const output of this.outputs) {
      handleOperation(output.operation);
    }
  }
}