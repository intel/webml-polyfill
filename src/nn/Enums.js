// This file is derived from NeuralNetworks.h of  https://android.googlesource.com/platform/frameworks/ml
// The license header of NeuralNetworks.h is:
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export const FuseCode = {
  /** NO fused activation function. */
  NONE: 0,
  /** Fused ReLU activation function. */
  RELU: 1,
  /** Fused ReLU1 activation function. */
  RELU1: 2,
  /** Fused ReLU6 activation function. */
  RELU6: 3,
};

export const OperandCode = {
  /** The following entries are used to declare scalars. */

  /** A 32 bit floating point scalar value. */
  FLOAT32: 0,
  /** A signed 32 bit integer scalar value. */
  INT32: 1,
  /** An unsigned 32 bit integer scalar value. */
  UINT32: 2,

  /** The following entries are used to declare tensors. */

  /** A tensor of 32 bit floating point values. */
  TENSOR_FLOAT32: 3,
  /** A tensor of 32 bit integer values. */
  TENSOR_INT32: 4,
  /** A tensor of 8 bit integers that represent real numbers.
   *
   * Attached to this tensor are two numbers that can be used to convert
   * the 8 bit integer to the real value and vice versa.  These two numbers are:
   * - scale: a 32 bit non-negative floating point value.
   * - zeroPoint: an 32 bit integer, in range [0, 255].
   *
   * The formula is:
   * real_value = (integer_value - zeroPoint) * scale.
   */
  TENSOR_QUANT8_ASYMM: 5,
};

export const PaddingCode = {
  /**
   * SAME padding.
   * Padding on both ends are the "same":
   *     padding_to_beginning =  total_padding / 2
   *     padding_to_end       = (total_padding + 1)/2.
   * i.e., for even number of padding, padding to both ends are exactly
   * the same; for odd number of padding, padding to the ending is bigger
   * than the padding to the beginning by 1.
   *
   * total_padding is a function of input, stride and filter size.
   * It could be computed as follows:
   *    out_size = (input + stride - 1) / stride;
   *    needed_input = (out_size - 1) * stride + filter_size
   *    total_padding = max(0, needed_input - output_size)
   *  The computation is the same for the horizontal and vertical directions.
   */
  SAME: 1,

  /**
   * VALID padding.
   * No padding. When the input size is not evenly divisible by
   * the filter size, the input at the end that could not fill
   * the whole filter tile will simply be ignored.
   */
  VALID: 2,
};

export const PreferenceCode = {
  /**
   * Prefer executing in a way that minimizes battery drain.
   * This is desirable for compilations that will be executed often.
   */
  LOW_POWER: 0,
  /**
   * Prefer returning a single answer as fast as possible, even if this causes
   * more power consumption.
   */
  FAST_SINGLE_ANSWER: 1,
  /**
   * Prefer maximizing the throughput of successive frames, for example when
   * processing successive frames coming from the camera.
   */
  SUSTAINED_SPEED: 2,
};

export const OperationCode = {
  /** Adds two tensors, element-wise.
   *
   * Takes two input tensors of identical type and compatible dimensions. The output
   * is the sum of both input tensors, optionally modified by an activation function.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the output is the maximum size along each dimension of the input operands.
   * It starts with the trailing dimensions, and works its way forward.
   *
   * Example:
   *
   *     input1.dimension = {4, 1, 2}
   *     input2.dimension = {5, 4, 3, 1}
   *     output.dimension = {5, 4, 3, 2}
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: A tensor of the same type, and compatible dimensions as input0.
   * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The sum, a tensor of the same type as input0.
   */
  ADD: 0,

  /** Performs a 2-D average pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sum_{i, j}(input[batch, row + i, col + j, channel]) / sum(1)
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" (i.e., Num_samples, Height, Width, and Channels)
   * data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 5: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 2: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the filter width.
   * * 5: An INT32 value, specifying the filter height.
   * * 6: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
   */
  AVERAGE_POOL_2D: 1,

  /** Concatenates the input tensors along the given dimension.
   *
   * The input tensors must have identical type and the same dimensions except the
   * dimension along the concatenation axis.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0 ~ n-1: The list of n input tensors, of shape [D0, D1, ..., Daxis(i), ..., Dm].
   *            For inputs of {@link TENSOR_QUANT8_ASYMM} type, all
   *            input tensors must have the same scale and zeroPoint.
   * * n: An INT32 value, specifying the concatenation axis.
   *
   * Outputs:
   * * 0: The output, a tensor of the same type as the input tensors.
   *      The output shape is [D0, D1, ..., sum(Daxis(i)), ..., Dm].
   */
  CONCATENATION: 2,

  /** Performs an 2-D convolution operation.
   *
   * The CONV_2D op sweeps a 2-D filter that can mix channels together over a batch of
   * images, applying the filter to each window of each image of the appropriate size.
   *
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sum_{i, j} (
   *             input[batch, row + i, col + j, k] *
   *             filter[channel, row + i, col + j, k] +
   *             bias[channel]
   *         )
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 4: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 6: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input_scale * filter_scale.
   */
  CONV_2D: 3,

  /** Performs a depthwise 2-D convolution operation.
   *
   * Given an input tensor of shape [batches, height, width, depth_in] and a filter
   * tensor of shape [1, filter_height, filter_width, depth_out] containing
   * depth_out convolutional filters of depth 1, DEPTHWISE_CONV applies a different
   * filter to each input channel (expanding from 1 channel to channel_multiplier channels
   * for each), then concatenates the results together.
   *
   * The output has depth_out = depth_in * depth_multiplier channels.
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[b, i, j, k * channel_multiplier + q] =
   *         sum_{di, dj} (
   *             input[b, strides[1] * i + di, strides[2] * j + dj, k] *
   *             filter[1, di, dj, k * channel_multiplier + q]
   *         )
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 9: An INT32 value, specifying the depthwise multiplier.
   * * 10: An INT32 value, and has to be one of the {@link FuseCode} values.
   *       Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 4: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the depthwise multiplier.
   * * 7: An INT32 value, and has to be one of the {@link FuseCode} values.
   *       Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input_scale * filter_scale.
   */
  DEPTHWISE_CONV_2D: 4,

  /** Rearranges data from depth into blocks of spatial data.
   *
   * More specifically, this op outputs a copy of the input tensor where values from
   * the depth dimension are moved in spatial blocks to the height and width dimensions.
   * The value block_size indicates the input block size and how the data is moved.
   *
   * Chunks of data of size block_size * block_size from depth are rearranged into
   * non-overlapping blocks of size block_size x block_size.
   *
   * The width of the output tensor is input_depth * block_size, whereas the height is
   * input_height * block_size.
   * The depth of the input tensor must be divisible by block_size * block_size
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
   *      block_size * block_size must be a divisor of the input depth.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batch, height*block_size, width*block_size,
   *      depth/(block_size*block_size)].
   */
  DEPTH_TO_SPACE: 5,

  /** Dequantizes the input tensor.
   *
   * The formula is:
   *
   *     output = (input - zeroPoint) * scale.
   *
   * Supported tensor types:
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor of type {@link TENSOR_QUANT8_ASYMM}.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0, but with type
   *      {@link TENSOR_FLOAT32}.
   */
  DEQUANTIZE: 6,

  /** Looks up sub-tensors in the input tensor.
   *
   * This operator takes for input a tensor of values (Values) and
   * a one-dimensional tensor of selection indices (Lookups).
   * The output tensor is the concatenation of sub-tensors of Values as
   * selected by Lookups.
   *
   * Think of Values as being sliced along its first dimension:
   * The entries in Lookups select which slices are concatenated together
   * to create the output tensor.
   *
   * For example, if Values has shape of [40, 200, 300] and
   * Lookups has shape of [3], we would expect all three values
   * found in Lookups to be  between 0 and 39. The resulting tensor will
   * have shape of [3, 200, 300].
   *
   * If a value in Lookups is out of bounds, the operation will fail
   * and an error will be reported.
   *
   * Inputs:
   * * 0: Lookups. A 1-D tensor of {@link TENSOR_INT32} type.
   *      The values are indices into the first dimension of Values.
   * * 1: Values. An n-D tensor, where n >= 2, from which sub-tensors are
   *      extracted.
   *
   * Output:
   * * 0: A n-D tensor with the same rank and shape as the Values
   *      tensor, except for the first dimension which has the same size
   *      as Lookups' only dimension.
   */
  EMBEDDING_LOOKUP: 7,

  /** Computes element-wise floor() on the input tensor.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   *
   * Outputs:
   * * 0: The output tensor, of the same type and dimensions as the input tensor.
   */
  FLOOR: 8,

  /** Denotes a fully (densely) connected layer, which connects all elements in the input
   * tensor with each element in the output tensor.
   *
   * This layer implements the operation:
   *
   *     outputs = activation(inputs * weights’ + bias)
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input. If rank is greater than 2, then it gets flattened to
   *      a 2-D Tensor. The 2-D Tensor is handled as if dimensions corresponded to shape
   *      [batch_size, input_size], where “batch_size” corresponds to the batching dimension,
   *      and “input_size” is the size of the input.
   * * 1: A 2-D tensor, specifying the weights, of shape [num_units, input_size], where
   *      "num_units" corresponds to the number of output nodes.
   * * 2: A 1-D tensor, of shape [num_units], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output tensor, of shape [batch_size, num_units].
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input_scale * filter_scale.
   */
  FULLY_CONNECTED: 9,

  /** Looks up sub-tensors in the input tensor using a key-value map.
   *
   * This operator takes for input a tensor of values (Values),
   * a one-dimensional tensor of selection values (Lookups) and
   * a one-dimensional tensor that maps these values to Values
   * indexes. The output tensor is the concatenation of sub-tensors of
   * Values as selected by Lookups via Keys.
   *
   * Think of Values as being sliced along its outer-most dimension.
   * The output is a concatenation of selected slices, with one slice
   * for each entry of Lookups. The slice selected is the one at the
   * same index as the Maps entry that matches the value in Lookups.
   *
   * For a hit, the corresponding sub-tensor of Values is included
   * in the Output tensor.  For a miss, the corresponding sub-tensor in
   * Output will have zero values.
   *
   * For example, if Values has shape of [40, 200, 300],
   * Keys should have a shape of [40]. If Lookups tensor has shape
   * of [3], we're concatenating three slices, so the resulting tensor
   * will have the shape of [3, 200, 300]. If the first entry in
   * Lookups has the value 123456, we'll look for that value in Keys tensor.
   * If the sixth entry of Keys contains 123456, we'll select the sixth
   * slice of Values. If no entry in Keys has 123456, a slice of zeroes
   * will be concatenated.
   *
   * Inputs:
   * * 0: Lookups. A 1-D {@link TENSOR_INT32} tensor with shape [ k ].
   * * 1: Keys. A 1-D {@link TENSOR_INT32} tensor with shape [ n ];
   *      Keys and Values pair represent a map, i.e., the ith element
   *      in Keys (Keys[i]) is the key to select the ith sub-tensor
   *      in Values (Values[i]), where 0 <= i <= n-1.
   *      Keys tensor *MUST* be sorted in ascending order.
   * * 2: Values. A tensor with shape of [ n, … ]; i.e., the first dimension must be n.
   *
   * Outputs:
   * * 0: Output. A tensor with shape [ k …].
   * * 1: Hits. A boolean tensor with shape [ k ] indicates whether the lookup
   *      hits (True) or not (False).
   *      Stored as {@link TENSOR_QUANT8_ASYMM} with offset 0 and scale 1.0f.
   *      A non-zero byte represents True, a hit. A zero indicates otherwise.
   */
  HASHTABLE_LOOKUP: 10,

  /** Applies L2 normalization along the depth dimension.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         input[batch, row, col, channel] /
   *         sqrt(sum_{c} pow(input[batch, row, col, c], 2))
   *
   * For input tensor with more dimensions, independently normalizes each 1-D slice along dimension dim.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout (i.e., Num_samples, Height, Width, and Channels).
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth].
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
   */
  L2_NORMALIZATION: 11,

  /** Performs an 2-D L2 pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         sqrt(sum_{i, j} pow(input[batch, row + i, col + j, channel], 2) / sum(1))
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 5: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 2: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the filter width.
   * * 5: An INT32 value, specifying the filter height.
   * * 6: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
   */
  L2_POOL_2D: 12,

  /** Applies Local Response Normalization along the depth dimension.
   *
   * The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last
   * dimension), and each vector is normalized independently. Within a given vector,
   * each component is divided by the weighted, squared sum of inputs within depth_radius.
   *
   * The output is calculated using this formula:
   *
   *     sqr_sum[a, b, c, d] =
   *         sum(pow(input[a, b, c, d - depth_radius : d + depth_radius + 1], 2)
   *     output = input / pow((bias + alpha * sqr_sum), beta)
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the radius of the normalization window.
   * * 2: A FLOAT32 value, specifying the bias, must not be zero.
   * * 3: A FLOAT32 value, specifying the scale factor, alpha.
   * * 4: A FLOAT32 value, specifying the exponent, beta.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  LOCAL_RESPONSE_NORMALIZATION: 13,

  /** Computes sigmoid activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = 1 / (1 + exp(-input))
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   *      For {@link TENSOR_QUANT8_ASYMM} type,
   *      the scale must be 1.f / 256 and the zeroPoint must be 0.
   */
  LOGISTIC: 14,

  /**
   * Projects an input to a bit vector via locality senstive hashing.
   *
   * Inputs:
   * * 0: Hash functions. Dim.size == 2, DataType: Float.
   *            Tensor[0].Dim[0]: Number of hash functions.
   *            Tensor[0].Dim[1]: Number of seeds per hash functions.
   *            Tensor[0].Dim[1] <= 32 in sparse case.
   *
   * * 1: Input. Dim.size >= 1, no restriction on DataType.
   * * 2: Weight. Optional. Dim.size == 1, DataType: Float.
   *     If not set, each input element is considered to have the same weight of
   *     1.0.
   *     Tensor[1].Dim[0] == Tensor[2].Dim[0]
   * * 3: Type:
   *        Sparse: Value LSHProjectionType_SPARSE(=1).
   *          Computed bit vector is considered to be sparse.
   *          Each output element is an int32 made up of multiple bits computed from
   *          hash functions.
   *
   *        Dense: Value LSHProjectionType_DENSE(=2).
   *          Computed bit vector is considered to be dense. Each output element
   *          represents a bit and can take the value of either 0 or 1.
   *
   * Outputs:
   * * 0: If the projection type is sparse:
   *        Output.Dim == { Tensor[0].Dim[0] }
   *        A tensor of int32 that represents hash signatures.
   *      If the projection type is Dense:
   *        Output.Dim == { Tensor[0].Dim[0] * Tensor[0].Dim[1] }
   *        A flattened tensor that represents projected bit vectors.
   */
  LSH_PROJECTION: 15,

  /**
   * Long short-term memory unit (LSTM) recurrent network layer.
   *
   * The default non-peephole implementation is based on:
   * http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
   * S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural
   * Computation, 9(8):1735-1780, 1997.
   *
   * The peephole implementation is based on:
   * https://research.google.com/pubs/archive/43905.pdf
   * Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory
   * recurrent neural network architectures for large scale acoustic modeling."
   * INTERSPEECH, 2014.
   *
   * The coupling of input and forget gate (CIFG) is based on:
   * http://arxiv.org/pdf/1503.04069.pdf
   * Greff et al. "LSTM: A Search Space Odyssey"
   *
   * The class has the following independently optional inputs:
   * * If input gate (if CIFG): “input_to_forget_weights”,
   *   “recurrent_to_input_weights”, “cell_to_input_weights”, “input_gate_bias”.
   * * If no peephole connections: “cell_to_input_weights”,
   *   “cell_to_forget_weights”, “cell_to_output_weights”.
   * * If no projection layer: “projection_weights” and “projection_bias”.
   * * If no projection bias: “projection_bias”.
   *
   * Supported tensor types (type T):
   * * {@link TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: Input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size”
   *      is the size of the input.
   * * 1: input_to_input_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of cell units.
   * * 2: input_to_forget_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 3: input_to_cell_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 4: input_to_output_weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size].
   * * 5: recurrent_to_input_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size], where
   *      “output_size” corresponds to either the number of cell units (i.e.,
   *      “num_units”), or the second dimension of the “projection_weights”, if
   *      defined.
   * * 6: recurrent_to_forget_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 7: recurrent_to_cell_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 8: recurrent_to_output_weights.
   *      A 2-D tensor of type T, of shape [num_units, output_size].
   * * 9: cell_to_input_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 10:cell_to_forget_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 11:cell_to_output_weights.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 12:input_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 13:forget_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 14:cell_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 15:output_gate_bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 16:projection_weights.
   *      A 2-D tensor of type T, of shape [output_size, num_units].
   * * 17:projection_bias.
   *      A 1-D tensor of type T, of shape [output_size].
   * * 18: output_state (in).
   *      A 2-D tensor of type T, of shape [batch_size, output_size].
   * * 19: cell_state (in).
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   * * 20:fused_activation_function.
   *      An optional {@link FuseCode} value indicating the activation
   *      function.
   *      If “NONE” is specified then it results in a linear activation.
   * * 21:cell_clip.
   *      A clipping threshold for the cell state, such that values are bound
   *      within [-cell_clip, cell_clip]. If set to 0.0 then clipping is
   *      disabled.
   * * 22:proj_clip.
   *      A clipping threshold for the output from the projection layer, such
   *      that values are bound within [-proj_clip, proj_clip]. If set to 0.0
   *      then clipping is disabled.
   *
   * Outputs:
   * * 0: scratch_buffer.
   *      A 3-D tensor of type T, of shape [batch_size, num_cell, 4].
   * * 1: output_state (out).
   *      A 2-D tensor of type T, of shape [batch_size, output_size].
   * * 2: cell_state (out).
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   * * 3: output.
   *      A 2-D tensor of type T, of shape [batch_size, output_size]. This is
   *      effectively the same as the current “output_state” value.
   */
  LSTM: 16,

  /** Performs an 2-D max pooling operation.
   *
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, row, col, channel] =
   *         max_{i, j} (input[batch, row + i, col + j, channel])
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 2: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 5: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 6: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the filter width.
   * * 8: An INT32 value, specifying the filter height.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 2: An INT32 value, specifying the stride when walking through input
   *      in the ‘width’ dimension.
   * * 3: An INT32 value, specifying the stride when walking through input
   *      in the ‘height’ dimension.
   * * 4: An INT32 value, specifying the filter width.
   * * 5: An INT32 value, specifying the filter height.
   * * 6: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth].
   */
  MAX_POOL_2D: 17,

  /** Multiplies two tensors, element-wise.
   *
   * Takes two input tensors of identical type and compatible dimensions. The output
   * is the product of both input tensors, optionally modified by an activation function.
   *
   * Two dimensions are compatible when:
   *     1. they are equal, or
   *     2. one of them is 1
   *
   * The size of the resulting output is the maximum size along each dimension of the
   * input operands. It starts with the trailing dimensions, and works its way forward.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4
   *
   * Inputs:
   * * 0: A tensor.
   * * 1: A tensor of the same type, and compatible dimensions as input0.
   * * 2: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The product, a tensor of the same type as input0.
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input1_scale * input2_scale.
   */
  MUL: 18,

  /** Computes rectified linear activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = max(0, input)
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  RELU: 19,

  /** Computes rectified linear 1 activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = min(1.f, max(-1.f, input))
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  RELU1: 20,

  /** Computes rectified linear 6 activation on the input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = min(6, max(0, input))
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  RELU6: 21,

  /** Reshapes a tensor.
   *
   * Given tensor, this operation returns a tensor that has the same values as tensor,
   * but with a newly specified shape.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the tensor to be reshaped.
   * * 1: A 1-D tensor of type {@link TENSOR_INT32}, defining the shape
   *      of the output tensor. The number of elements implied by shape must be the same
   *      as the number of elements in the input tensor.
   *
   * Outputs:
   * * 0: The output tensor, of shape specified by the input shape.
   */
  RESHAPE: 22,

  /** Resizes images to given size using the bilinear interpretation.
   *
   * Support align_corners parameter. Default value is FALSE (0).
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs (without align_corners):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the output height of the output tensor.
   * * 2: An INT32 value, specifying the output width of the output tensor.
   *
   * Inputs (with align_corners):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth], specifying the input.
   * * 1: An INT32 value, specifying the output height of the output tensor.
   * * 2: An INT32 value, specifying the output width of the output tensor.
   * * 3: An INT32 value, specifying align_corners parameter. If TRUE (1), the centers of
   *      the 4 corner pixels of the input and output tensors are aligned, preserving
   *      the values at the corner pixels.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, new_height, new_width, depth].
   */
  RESIZE_BILINEAR: 23,

  /**
   * A basic recurrent neural network layer.
   *
   * This layer implements the operation:
   * outputs = state = activation(inputs * input_weights + state * recurrent_weights + bias)
   *
   * Where:
   * * “input_weights” is a weight matrix that multiplies the inputs;
   * * “recurrent_weights” is a weight matrix that multiplies the current
   *    “state” which itself is the output from the previous time step
   *    computation;
   * * “bias” is a bias vector (added to each output vector in the batch);
   * * “activation” is the function passed as the “fused_activation_function”
   *   argument (if not “NONE”).
   *
   * Supported tensor types (Type T):
   * * {@link TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size” is
   *      the size of the input.
   * * 1: weights.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of units.
   * * 2: recurrent_weights.
   *      A 2-D tensor of type T, of shape [num_units, num_units], with columns
   *      corresponding to the weights from each unit.
   * * 3: bias.
   *      A 1-D tensor of type T, of shape [num_units].
   * * 4: hidden state (in).
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   * * 5: fused_activation_function.
   *      An optional {@link FuseCode} value indicating the activation
   *      function. If “NONE” is specified then it results in a linear
   *      activation.
   *
   * Outputs:
   * * 0: hidden state (out).
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   *
   * * 1: output.
   *      A 2-D tensor of type T, of shape [batch_size, num_units]. This is
   *      effectively the same as the current state value.
   */
  RNN: 24,

  /** Computes the softmax activation on the input tensor element-wise, per batch, by
   * normalizing the input vector so the maximum coefficient is zero.
   *
   * The output is calculated using this formula:
   *
   *     output[batch, i] =
   *         exp((input[batch, i] - max(input[batch, :])) * beta) /
   *         sum_{k}{exp((input[batch, k] - max(input[batch, :])) * beta)}
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 2 or 4.
   *
   * Inputs:
   * * 0: A 2-D or 4-D tensor, specifying the tensor to be reshaped.
   * * 1: A FLOAT32 value, specifying the positive scaling factor for the exponent, beta.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   *      For {@link TENSOR_QUANT8_ASYMM} type,
   *      the scale must be 1.f / 256 and the zeroPoint must be 0.
   */
  SOFTMAX: 25,

  /** Rearranges blocks of spatial data, into depth.
   *
   * More specifically, this op outputs a copy of the input tensor where values from
   * the height and width dimensions are moved to the depth dimension.
   * The value block_size indicates the input block size and how the data is moved.
   *
   * Chunks of data of size block_size * block_size from depth are rearranged into
   * non-overlapping blocks of size block_size x block_size.
   *
   * The depth of the output tensor is input_depth * block_size * block_size.
   * The input tensor's height and width must be divisible by block_size.
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Inputs:
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: An INT32 value, specifying the block_size. block_size must be >=1 and
   *      block_size must be a divisor of both the input height and width.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batch, height/block_size, width/block_size,
   *      depth*block_size*block_size].
   */
  SPACE_TO_DEPTH: 26,

  /**
   * SVDF op is a kind of stateful layer derived from the notion that a
   * densely connected layer that's processing a sequence of input frames can
   * be approximated by using a singular value decomposition of each of its
   * nodes. The implementation is based on:
   *
   * https://research.google.com/pubs/archive/43813.pdf
   *
   * P. Nakkiran, R. Alvarez, R. Prabhavalkar, C. Parada.
   * “Compressing Deep Neural Networks using a Rank-Constrained Topology”.
   * INTERSPEECH, 2015.
   *
   * It processes the incoming input using a 2-stage filtering mechanism:
   * * stage 1 performs filtering on the "features" dimension, whose outputs get
   *   pushed into a memory of fixed-size memory_size.
   * * stage 2 performs filtering on the "time" dimension of the memory_size
   *   memoized outputs of stage 1.
   *
   * Specifically, for rank 1, this layer implements the operation:
   *
   *    memory = push(conv1d(inputs, weights_feature, feature_dim,
   *                  "PADDING_VALID"));
   *    outputs = activation(memory * weights_time + bias);
   *
   * Where:
   * * “weights_feature” is a weights matrix that processes the inputs (by
   *   convolving the input with every “feature filter”), and whose outputs get
   *   pushed, stacked in order, into the fixed-size “memory” (the oldest entry
   *   gets dropped);
   * * “weights_time” is a weights matrix that processes the “memory” (by a
   *   batched matrix multiplication on the num_units);
   * * “bias” is an optional bias vector (added to each output vector in the
   *   batch); and
   * * “activation” is the function passed as the “fused_activation_function”
   *   argument (if not “NONE”).
   *
   * Each rank adds a dimension to the weights matrices by means of stacking
   * the filters.
   *
   * Supported tensor types (type T):
   * * {@link TENSOR_FLOAT32}
   *
   * Inputs:
   * * 0: input.
   *      A 2-D tensor of type T, of shape [batch_size, input_size], where
   *      “batch_size” corresponds to the batching dimension, and “input_size” is
   *      the size of the input.
   * * 1: weights_feature.
   *      A 2-D tensor of type T, of shape [num_units, input_size], where
   *      “num_units” corresponds to the number of units.
   * * 2: weights_time.
   *      A 2-D tensor of type T, of shape [num_units, memory_size], where
   *      “memory_size” corresponds to the fixed-size of the memory.
   * * 3: bias.
   *      An optional 1-D tensor of type T, of shape [num_units].
   * * 4: state (in).
   *      A 2-D tensor of type T, of shape [batch_size, (memory_size - 1) * num_units * rank].
   * * 5: rank.
   *      The rank of the SVD approximation.
   * * 6: fused_activation_function.
   *      An optional {@link FuseCode} value indicating the activation function.
   *      If “NONE” is specified then it results in a linear activation.
   *
   * Outputs:
   * * 0: state (out).
   *      A 2-D tensor of type T, of shape [batch_size, (memory_size - 1) * num_units * rank].
   * * 1: output.
   *      A 2-D tensor of type T, of shape [batch_size, num_units].
   */
  SVDF: 27,

  /** Computes hyperbolic tangent of input tensor element-wise.
   *
   * The output is calculated using this formula:
   *
   *     output = tanh(input)
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   *
   * Supported tensor rank: up to 4.
   *
   * Inputs:
   * * 0: A tensor, specifying the input.
   *
   * Outputs:
   * * 0: The output tensor of same shape as input0.
   */
  TANH: 28,

  /** Performs a atrous 2-D convolution operation.
   *
   * The ATROUS_CONV_2D op sweeps a 2-D filter that can mix channels together over a batch of
   * images, applying the filter to each window of each image of the appropriate size.
   *
   * If the dilation rate parameters are greater than one, it performs convolution with holes,
   * sampling the input values every rate pixels in the height and width dimensions.
   *
   * The output dimensions are functions of the filter dimensions, stride, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[batch, height, width, out_channel] =
   *        sum_{dheight, dwidth, in_channel} (
   *          filters[dheight, dwidth, in_channel, out_channel] *
   *          value[batch, height + rate*dheight, width + rate*dwidth, in_channel]
   *        )
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the dilation rate in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the dilation rate in the ‘height’ dimension.
   * * 9: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [depth_out, filter_height, filter_width, depth_in],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 4: An INT32 value, specifying the dilation rate in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the dilation rate in the ‘height’ dimension.
   * * 6: An INT32 value, and has to be one of the {@link FuseCode} values.
   *      Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input_scale * filter_scale.
   */
  ATROUS_CONV_2D: 10003,

  /** Performs a atrous depthwise 2-D convolution operation.
   *
   * Given an input tensor of shape [batches, height, width, depth_in] and a filter
   * tensor of shape [1, filter_height, filter_width, depth_out] containing
   * depth_out convolutional filters of depth 1, DEPTHWISE_CONV applies a different
   * filter to each input channel (expanding from 1 channel to channel_multiplier channels
   * for each), then concatenates the results together.
   *
   * If the dilation rate parameters are greater than one, it performs convolution with holes,
   * sampling the input values every rate pixels in the height and width dimensions.
   *
   * The output has depth_out = depth_in * depth_multiplier channels.
   * The output dimensions are functions of the filter dimensions, dilation rate, and padding.
   *
   * The values in the output tensor are computed as:
   *
   *     output[b, i, j, k * channel_multiplier + q] = sum_{di, dj}
   *         filter[di, dj, k, q] * input[b, i + rate[0] * di,
   *                                         j + rate[1] * dj, k]
   *
   * Supported tensor types:
   * * {@link TENSOR_FLOAT32}
   * * {@link TENSOR_QUANT8_ASYMM}
   *
   * Supported tensor rank: 4, with "NHWC" data layout.
   *
   * Both explicit padding and implicit padding are supported.
   *
   * Inputs (explicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the padding on the left, in the ‘width’ dimension.
   * * 4: An INT32 value, specifying the padding on the right,in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the padding on the top, in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the padding on the bottom, in the ‘height’ dimension.
   * * 7: An INT32 value, specifying the dilation rate in the ‘width’ dimension.
   * * 8: An INT32 value, specifying the dilation rate in the ‘height’ dimension.
   * * 9: An INT32 value, specifying the depthwise multiplier.
   * * 10: An INT32 value, and has to be one of the {@link FuseCode} values.
   *       Specifies the activation to invoke on the result of each addition.
   *
   * Inputs (implicit padding):
   * * 0: A 4-D tensor, of shape [batches, height, width, depth_in], specifying the input.
   * * 1: A 4-D tensor, of shape [1, filter_height, filter_width, depth_out],
   *      specifying the filter.
   * * 2: A 1-D tensor, of shape [depth_out], specifying the bias.
   *      For input tensor of {@link TENSOR_FLOAT32} type, the bias should
   *      also be of {@link TENSOR_FLOAT32}.
   *      For input tensor of {@link TENSOR_QUANT8_ASYMM} type, the bias
   *      should be of {@link TENSOR_INT32}, with zeroPoint of 0 and
   *      bias_scale == input_scale * filter_scale.
   * * 3: An INT32 value, specifying the implicit padding scheme, has to be one of the
   *      {@link PaddingCode} values.
   * * 4: An INT32 value, specifying the dilation rate in the ‘width’ dimension.
   * * 5: An INT32 value, specifying the dilation rate in the ‘height’ dimension.
   * * 6: An INT32 value, specifying the depthwise multiplier.
   * * 7: An INT32 value, and has to be one of the {@link FuseCode} values.
   *       Specifies the activation to invoke on the result of each addition.
   *
   * Outputs:
   * * 0: The output 4-D tensor, of shape [batches, out_height, out_width, depth_out].
   *      For output tensor of {@link TENSOR_QUANT8_ASYMM} type, the following
   *      condition must be satisfied: output_scale > input_scale * filter_scale.
   */
  ATROUS_DEPTHWISE_CONV_2D: 10004,
};

export const ResultCode = {
  NO_ERROR: 0,
  OUT_OF_MEMORY: 1,
  INCOMPLETE: 2,
  UNEXPECTED_NULL: 3,
  BAD_DATA: 4,
  OP_FAILED: 5,
  UNMAPPABLE: 5,
  BAD_STATE: 6,
};

export const OperandLifetime = {
  TEMPORARY_VARIABLE: 0,
  MODEL_INPUT: 1,
  MODEL_OUTPUT: 2,
  CONSTANT_COPY: 3,
  CONSTANT_REFERENCE: 4,
  NO_VALUE: 5
};
