import { TypedArray, createTensor, validateTypedArray } from './utils';
import { Input } from './Input';
import { Model } from './Model';

import * as tf from '@tensorflow/tfjs-core'

export class Execution {
  private model_: Model; 
  private inputTensors_: Map<Input, tf.Tensor> = new Map();
  private outputBuffers_: Array<TypedArray> = [];

  constructor(model: Model) {
    this.model_ = model;
  }

  setInput(index: number, data: TypedArray): void {
    const input = this.model_.inputs_[index];
    validateTypedArray(data, input.desc.type);
    this.inputTensors_.set(input, createTensor(input.desc, data));
  }

  setOutput(index: number, data: TypedArray): void {
    // TODO: validate typed array
    this.outputBuffers_[index] = data;
  }

  async startCompute(): Promise<void> {
    for (let i = 0; i < this.model_.outputs_.length; ++i) {
      const tensor: tf.Tensor = this.model_.outputs_[i].operation.run(this);
      const data = await tensor.data();
      this.outputBuffers_[i].set(data);
    }
  }

  getInputTensor(input: Input): tf.Tensor {
    return this.inputTensors_.get(input);
  }
}