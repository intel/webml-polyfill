import { Output } from './Output';
import { Input } from './Input';
import { Constant } from './Constant';
import { Operation } from './Operation';
import { CompilationOptions } from './CompilationOptions';
import { Compilation } from './Compilation';
import { assert } from './utils';
import { NamedOperand } from './NamedOperand';

export class Model {
  inputs_: Map<string, Input> = new Map();
  outputs_: Map<string, Output> = new Map();
  constants_: Array<Constant> = [];

  constructor(outputs: Array<NamedOperand>) {
    assert(outputs.length !== 0, 'The length of outputs parameter should not be 0.');
    assert(outputs.every(namedOutput => typeof namedOutput.name === 'string' &&
        namedOutput.operand instanceof Output), 'The outputs parameter is invalid.');
    for (const namedOutput of outputs) {
      this.outputs_.set(namedOutput.name, namedOutput.operand as Output);
    }
    this.initialize();
  }

  async createCompilation(options: CompilationOptions): Promise<Compilation> {
    const compilation = new Compilation(options, this);
    await compilation.compile();
    return compilation;
  }

  private initialize(): void {
    const self = this;
    function handleOperation(operation: Operation): void {
      for (const operand of operation.inputs) {
        if (operand instanceof Input) {
          const input = operand as Input;
          self.inputs_.set(input.name, input);
        } else if (operand instanceof Constant) {
          self.constants_.push((operand as Constant));
        } else if (operand instanceof Output) {
          handleOperation((operand as Output).operation);
        }
      }
    }
    for (const output of this.outputs_.values()) {
      handleOperation(output.operation);
    }
  }
}