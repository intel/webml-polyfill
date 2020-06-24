import { TypedArray, createTensor, validateTypedArray } from './utils';
import { Input } from './Input';
import { Compilation } from './Compilation';
import { ExecutionContext } from './ExecutionContext';

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
    validateTypedArray(data, input.desc);
    this.inputTensors_.set(input, createTensor(input.desc, data));
  }

  setOutput(index: number, data: TypedArray): void {
    const output = this.compilation_.model_.outputs_[index];
    const desc = this.compilation_.outputDescriptors_.get(output);
    validateTypedArray(data, desc)
    this.outputBuffers_[index] = data;
  }

  async startCompute(): Promise<void> {
    for (let i = 0; i < this.compilation_.model_.outputs_.length; ++i) {
      const tensor: tf.Tensor = tf.tidy(() => {
        return this.compilation_.model_.outputs_[i].operation.run({
          inputTensors: this.inputTensors_,
          constantTenosrs: this.compilation_.constantTensors_
        } as ExecutionContext);
      });
      const data = await tensor.data();
      tf.dispose(tensor);
      this.outputBuffers_[i].set(data);
    }
    for (const tensor of this.inputTensors_.values()) {
      tf.dispose(tensor);
    }
  }
}