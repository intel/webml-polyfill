import { Binary } from "./Binary";

import * as tf from '@tensorflow/tfjs-core';

export class Add extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.add(a, b);
  }
}