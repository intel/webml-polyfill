import { Operation } from "../Operation";
import { Output } from "../Output";
import { Operand } from "../Operand";
import { ExecutionContext } from "../ExecutionContext";

import * as tf from '@tensorflow/tfjs-core';

export abstract class Binary extends Operation {
  constructor(a: Operand, b: Operand) {
    super([a, b]);
    this.outputs.push(new Output(this));
  }

  run(context: ExecutionContext): tf.Tensor {
    const a: tf.Tensor = this.getTensor(this.inputs[0], context);
    const b: tf.Tensor = this.getTensor(this.inputs[1], context);
    return this.runOp(a, b);
  }

  abstract runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor;
}