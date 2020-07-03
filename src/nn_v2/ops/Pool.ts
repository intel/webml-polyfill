import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { OperandLayout } from "../OperandLayout";
import { ExecutionContext } from "../ExecutionContext";
import { assert, isNumberArray } from "../utils";

import * as tf from '@tensorflow/tfjs-core';

export abstract class Pool extends Operation {
  protected windowDimensions_: [number, number];
  protected padding_: [number, number, number, number];
  protected strides_: [number, number];
  protected dilations_: [number, number];
  protected groups_: number;
  protected layout_: OperandLayout;

  constructor(input: Operand,
              windowDimensions: [number, number] = [-1, -1],
              padding: [number, number, number, number] = [0, 0, 0, 0],
              strides: [number, number] = [1, 1],
              dilations: [number, number] = [1, 1],
              layout: OperandLayout = OperandLayout.nchw) {
    super([input]);

    assert(isNumberArray(windowDimensions) && windowDimensions.length === 2, 'The padding parameter is invalid.');
    this.windowDimensions_ = windowDimensions;

    assert(isNumberArray(padding) && padding.length === 4, 'The padding parameter is invalid.');
    this.padding_ = padding;

    assert(isNumberArray(strides) && strides.length === 2, 'The strides parameter is invalid.');
    this.strides_ = strides;

    assert(isNumberArray(dilations) && dilations.length === 2, 'The dilations parameter is invalid.');
    this.dilations_ = dilations;

    assert(layout in OperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D = this.getTensor(this.inputs[0], context) as tf.Tensor4D;
    assert(this.padding_.every(v => v === this.padding_[0]), 'The tf.conv2d only supports the same padding value.');
    const padding = this.padding_[0];
    const poolingType = this.getPoolingType();
    if (this.layout_ === OperandLayout.nchw) {
      // nchw -> nhwc
      input = input.transpose([0, 2, 3, 1]);
    }
    let windowDimensions = this.windowDimensions_;
    if (windowDimensions[0] === -1 && windowDimensions[1] === -1) {
      windowDimensions[0] = input.shape[1];
      windowDimensions[1] = input.shape[2];
    }
    let output = tf.pool(input, this.windowDimensions_, poolingType, padding, this.dilations_, this.strides_);
    if (this.layout_ === OperandLayout.nchw) {
      // nhwc -> nchw
      output = output.transpose([0, 3, 1, 2]);
    }
    return output;
  }

  abstract getPoolingType(): 'avg'|'max';
}