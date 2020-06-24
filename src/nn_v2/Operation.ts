import { Output } from "./Output";
import { Execution } from './Execution'
import { Operand } from "./Operand";
import { Constant } from "./Constant";
import { Input } from "./Input";

import * as tf from '@tensorflow/tfjs-core'

export abstract class Operation {
  inputs: Array<Operand> = [];
  outputs: Array<Output> = [];

  constructor(inputs: Array<Operand>) {
    this.inputs = inputs;
  }

  get output(): Output {
    return this.outputs[0];
  }

  protected getTensor(operand: Operand, execution: Execution): tf.Tensor {
    if (operand instanceof Constant) {
      return execution.getConstantTensor(operand as Constant);
    } else if (operand instanceof Output) {
      return (operand as Output).operation.run(execution);
    } else if (operand instanceof Input) {
      return execution.getInputTensor(operand as Input);
    } else {
      throw new Error('The operand is invalid.');
    }
  }

  abstract run(execution: Execution): tf.Tensor;
}