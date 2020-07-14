import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { ExecutionContext } from "../ExecutionContext";

import * as tf from '@tensorflow/tfjs-core';

export class Softmax extends Operation {
  constructor(x: Operand) {
    super([x]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const x: tf.Tensor = this.getTensor(this.inputs[0], context);
    if (x.rank !== 2) {
      throw new Error('The rank of x parameter should be 2.');
    }
    return tf.softmax(x);
  }
}