import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { OperandLayout } from "../OperandLayout";
import { ExecutionContext } from "../ExecutionContext";
import { assert, isNumberArray, isNumber } from "../utils";

import * as tf from '@tensorflow/tfjs-core'


export class Conv2d extends Operation {
  private padding_: [number, number, number, number];
  private strides_: [number, number];
  private dilations_: [number, number];
  private groups_: number;
  private layout_: OperandLayout;

  constructor(input: Operand, filter: Operand,
              padding: [number, number, number, number] = [0, 0, 0, 0],
              strides: [number, number] = [1, 1],
              dilations: [number, number] = [1, 1],
              groups: number = 1,
              layout: OperandLayout = OperandLayout.nchw) {
    super([input, filter]);

    assert(isNumberArray(padding) && padding.length === 4, 'The padding parameter is invalid.');
    this.padding_ = padding;

    assert(isNumberArray(strides) && strides.length === 2, 'The strides parameter is invalid.');
    this.strides_ = strides;

    assert(isNumberArray(dilations) && dilations.length === 2, 'The dilations parameter is invalid.');
    this.dilations_ = dilations;

    assert(isNumber(groups), 'The gourps parameter is invalid.');
    this.groups_ = groups;

    assert(layout in OperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D = this.getTensor(this.inputs[0], context) as tf.Tensor4D;
    let filter: tf.Tensor4D = this.getTensor(this.inputs[1], context) as tf.Tensor4D;
    assert(this.padding_.every(v => v === this.padding_[0]), 'The tf.conv2d only supports the same padding value.');
    const padding = this.padding_[0];
    let input_channels:number;
    if (this.layout_ === OperandLayout.nchw) {
      // nchw -> nhwc
      input = input.transpose([0, 2, 3, 1]);
      input_channels = input.shape[1];
      // webnn layout: [output_channels, input_channels/groups, height, width] ->
      // tf.js layout: [filterHeight, filterWidth, inDepth, outDepth]
      filter = filter.transpose([2, 3, 1, 0]);
    } else {
      // 'NHWC'
      input_channels = input.shape[3];
    }
    let output;
    if (this.groups_ === 1) {
      output = tf.conv2d(input, filter, this.strides_, padding, 'NHWC', this.dilations_);
    } else if (this.groups_ === input_channels) {
      output = tf.depthwiseConv2d(input, filter, this.strides_, padding, 'NHWC', this.dilations_);
    } else {
      throw new Error(`The tf.js convolution doesn't support groups parameter ${this.groups_}`);
    }
    if (this.layout_ === OperandLayout.nchw) {
      // nhwc -> nchw
      output = output.transpose([0, 3, 1, 2]);
    }
    return output;
  }
}