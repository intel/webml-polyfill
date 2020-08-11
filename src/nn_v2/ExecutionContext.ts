import { Input } from './Input';
import { Constant } from './Constant';

import * as tf from '@tensorflow/tfjs-core'

export interface ExecutionContext {
  inputTensors: Map<Input, tf.Tensor>;
  constantTenosrs: Map<Constant, tf.Tensor>;
}