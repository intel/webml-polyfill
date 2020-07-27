import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { ExecutionContext } from "../ExecutionContext";
import { assert, isNumberArray } from "../utils";

import * as tf from '@tensorflow/tfjs-core';

export class Transpose extends Operation {
  private permutation_: number[];

  constructor(input: Operand, permutation?: number[]) {
    super([input]);
    if (permutation) {
      assert(isNumberArray(permutation), 'The permutation parameter is invalid.');
      this.permutation_ = permutation;
    } else {
      this.permutation_ = undefined;
    }
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.transpose(input, this.permutation_);
  }
}