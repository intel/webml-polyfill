import { assert, TypedArray, createTensor, validateTypedArray } from './utils';
import { Input } from './Input';
import { Output } from './Output';
import { Compilation } from './Compilation';
import { ExecutionContext } from './ExecutionContext';

import * as tf from '@tensorflow/tfjs-core'

export class Execution {
  private compilation_: Compilation;
  private inputTensors_: Map<Input, tf.Tensor> = new Map();
  private outputBuffers_: Map<Output, TypedArray> = new Map();

  constructor(compilation: Compilation) {
    this.compilation_ = compilation;
  }

  setInput(name: string, data: TypedArray): void {
    assert(typeof name === 'string' &&
        this.compilation_.model_.inputs_.has(name), 'The name parameter is invalid.');
    const input = this.compilation_.model_.inputs_.get(name);
    validateTypedArray(data, input.desc);
    this.inputTensors_.set(input, createTensor(input.desc, data));
  }

  setOutput(name: string, data: TypedArray): void {
    assert(typeof name === 'string' &&
        this.compilation_.model_.outputs_.has(name), 'The name parameter is invalid.');
    const output = this.compilation_.model_.outputs_.get(name);
    const desc = this.compilation_.outputDescriptors_.get(output);
    validateTypedArray(data, desc)
    this.outputBuffers_.set(output, data);
  }

  async startCompute(): Promise<void> {
    for (const output of this.compilation_.model_.outputs_.values()) {
      const tensor: tf.Tensor = tf.tidy(() => {
        return output.operation.run({
          inputTensors: this.inputTensors_,
          constantTenosrs: this.compilation_.constantTensors_
        } as ExecutionContext);
      });
      const data = await tensor.data();
      tf.dispose(tensor);
      this.outputBuffers_.get(output).set(data);
    }
    for (const tensor of this.inputTensors_.values()) {
      tf.dispose(tensor);
    }
  }
}