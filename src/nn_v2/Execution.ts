import { TypedArray, createTensor, validateTypedArray } from './utils';
import { Input } from './Input';
import { Constant } from './Constant';
import { Compilation } from './Compilation';

import * as tf from '@tensorflow/tfjs-core'

export class Execution {
  private compilation_: Compilation;
  private inputTensors_: Map<Input, tf.Tensor> = new Map();
  private outputBuffers_: Array<TypedArray> = [];

  constructor(compilation: Compilation) {
    this.compilation_ = compilation;
  }

  setInput(index: number, data: TypedArray): void {
    const input = this.compilation_.model_.inputs_[index];
    validateTypedArray(data, input.desc.type);
    this.inputTensors_.set(input, createTensor(input.desc, data));
  }

  setOutput(index: number, data: TypedArray): void {
    // TODO: validate typed array
    this.outputBuffers_[index] = data;
  }

  async startCompute(): Promise<void> {
    for (let i = 0; i < this.compilation_.model_.outputs_.length; ++i) {
      const tensor: tf.Tensor = tf.tidy(() => {
        return this.compilation_.model_.outputs_[i].operation.run(this);
      });
      const data = await tensor.data();
      tf.dispose(tensor);
      this.outputBuffers_[i].set(data);
    }
    for (let tensor of this.inputTensors_.values()) {
      tf.dispose(tensor);
    }
  }

  getInputTensor(input: Input): tf.Tensor {
    return this.inputTensors_.get(input);
  }

  getConstantTensor(constant: Constant): tf.Tensor {
    return this.compilation_.constantTensors_.get(constant);
  }
}