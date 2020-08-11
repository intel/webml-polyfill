import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { ExecutionContext } from "../ExecutionContext";

import * as tf from '@tensorflow/tfjs-core';

export class Relu extends Operation {
  constructor(input: Operand) {
    super([input]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.relu(input);
  }
}