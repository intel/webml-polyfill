import { Output } from "./Output";
import { Operand } from "./Operand";
import { Constant } from "./Constant";
import { Input } from "./Input";
import { ExecutionContext } from "./ExecutionContext";
import { assert } from "./utils";

import * as tf from '@tensorflow/tfjs-core'

export abstract class Operation {
  inputs: Array<Operand> = [];
  outputs: Array<Output> = [];

  constructor(inputs: Array<Operand>) {
    assert(inputs.every(input => input instanceof Operand), 'The inputs parameter is invalid.');
    this.inputs = inputs;
    this.outputs.push(new Output(this));
  }

  get output(): Output {
    return this.outputs[0];
  }

  protected getTensor(operand: Operand, context: ExecutionContext): tf.Tensor {
    if (operand instanceof Constant) {
      return context.constantTenosrs.get(operand as Constant);
    } else if (operand instanceof Input) {
      return context.inputTensors.get(operand as Input);
    } else if (operand instanceof Output) {
      return (operand as Output).operation.run(context);
    } else {
      throw new Error('The operand is invalid.');
    }
  }

  abstract run(context: ExecutionContext): tf.Tensor;
}