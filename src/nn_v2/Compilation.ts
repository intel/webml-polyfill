import { CompilationOptions } from "./CompilationOptions";
import { Execution } from "./Execution";
import { Model } from "./Model";
import { Constant } from "./Constant";

import * as tf from '@tensorflow/tfjs-core'

export class Compilation {
  model_: Model;
  constantTensors_: Map<Constant, tf.Tensor> = new Map();

  constructor(options: CompilationOptions, model: Model) {
    // TODO: support compilation options.
    this.model_ = model;
  }

  async compile(): Promise<void> {
    if (!(await tf.setBackend('webgl'))) {
      throw new Error('Failed to set tf.js webgl backend.');
    }
    await tf.ready();
    for (let constant of this.model_.constants_) {
      this.constantTensors_.set(constant, constant.createTensor());
    }
  }

  async createExecution(): Promise<Execution> {
    return new Execution(this);
  }
}