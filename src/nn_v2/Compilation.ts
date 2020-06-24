import { CompilationOptions } from "./CompilationOptions";
import { Execution } from "./Execution";
import { ExecutionContext } from "./ExecutionContext";
import { Model } from "./Model";
import { Constant } from "./Constant";
import { Input } from './Input';
import { getTypedArray, sizeFromDimensions, createTensor, createOperandDescriptorFromTensor } from "./utils";
import { Output } from "./Output";
import { OperandDescriptor } from "./OperandDescriptor";

import * as tf from '@tensorflow/tfjs-core'

export class Compilation {
  model_: Model;
  constantTensors_: Map<Constant, tf.Tensor> = new Map();
  outputDescriptors_: Map<Output, OperandDescriptor> = new Map();

  constructor(options: CompilationOptions, model: Model) {
    // TODO: support compilation options.
    this.model_ = model;
  }

  async compile(): Promise<void> {
    if (!(await tf.setBackend('webgl'))) {
      throw new Error('Failed to set tf.js webgl backend.');
    }
    await tf.ready();
    this.allocateConstants_();
    await this.inferOutputShapes_();
  }

  allocateConstants_ () {
    for (const constant of this.model_.constants_) {
      this.constantTensors_.set(constant, constant.createTensor());
    }
  }

  async inferOutputShapes_() {
    const inputTensors:  Map<Input, tf.Tensor> = new Map();
    for (const input of this.model_.inputs_) {
      const typedArrayConstructor = getTypedArray(input.desc.type);
      const inputBuffer = new typedArrayConstructor(sizeFromDimensions(input.desc.dimensions));
      inputTensors.set(input, createTensor(input.desc, inputBuffer));
    }
    for (const output of this.model_.outputs_) {
      const tensor: tf.Tensor = tf.tidy(() => {
        return output.operation.run({
          inputTensors: inputTensors,
          constantTenosrs: this.constantTensors_
        } as ExecutionContext);
      });
      await tensor.data();
      this.outputDescriptors_.set(output, createOperandDescriptorFromTensor(tensor));
    }
    for (const tensor of inputTensors.values()) {
      tf.dispose(tensor);
    }
  }

  async createExecution(): Promise<Execution> {
    return new Execution(this);
  }
}