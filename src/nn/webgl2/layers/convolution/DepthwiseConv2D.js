import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../webgl/fragmentShader/activation'
import webgl2 from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import ops from 'ndarray-ops'
import matMulDepthwiseShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/matMulDepthwise'
import depthwiseConv2DShaderSourceFunc from '../../webgl/fragmentShader/convolution/depthwiseConv2D'
import { fuseShaderSource } from '../../webgl/fragmentShader/activation/fuse'

/**
 * DepthwiseConv2D layer class
 */
export default class DepthwiseConv2D extends Layer {
  /**
   * Creates a DepthwiseConv2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'DepthwiseConv2D';
    const {
      filters = 1,
      kernel_size = [3, 3],
      strides = [1, 1],
      padding = 'VALID',
      depthMultiplier = 1,
      data_format = 'NHWC',
      dilation_rate = [1, 1],
      activation = 'NONE',
      use_bias = true,
      weights = []
    } = attrs;

    if (Array.isArray(kernel_size)) {
      this.kernelShape = [filters, ...kernel_size];
    } else {
      this.kernelShape = [filters, kernel_size, kernel_size];
    }

    if (Array.isArray(strides)) {
      this.strides = strides;
    } else {
      this.strides = [strides, strides];
    }

    if (Array.isArray(padding)) {
      if (padding.length !== 4) {
        this.throwError('Invalid padding.');
        // If all numbers in padding are 0, use padding = 'VALID'
      } else if (padding.every((x)=>!x)) {
        this.padding = 'VALID';
      } else {
        this.padding = padding;
      }
    } else if (padding === 'VALID' || padding === 'SAME') {
      this.padding = padding;
    } else {
      this.throwError('Invalid padding.');
    }

    if (data_format === 'NHWC' || data_format === 'NCHW') {
      this.dataFormat = data_format;
    } else {
      this.throwError('Only NHWC and NCHW data formats are allowed.');
    }

    if (Array.isArray(dilation_rate)) {
      this.dilationRate = dilation_rate;
    } else {
      this.dilationRate = [dilation_rate, dilation_rate];
    }
    if (
      (this.dilationRate[0] !== 1 || this.dilationRate[1] !== 1) &&
      (this.strides[0] !== 1 || this.strides[1] !== 1)
    ) {
      // Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1
      this.throwError(`Incompatible combination of dilation_rate with strides.`);
    }

    this.depthMultiplier = depthMultiplier;
    this.activation = activation;
    this.fuseActivation = fuseShaderSource[this.activation];

    this.useBias = use_bias;
    this._setWeights(weights);
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   * 
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  _setWeights(weightsArr) {
    if (this.dataFormat === 'NCHW') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(0, 2, 3, 1);
    }
    
    const params = this.useBias ? ['kernel', 'bias'] : ['kernel'];
    const inputChannels = weightsArr[0].tensor.shape[3];
    const [filter, kernelH, kernelW] = this.kernelShape;

    super.setWeights(params, weightsArr, false);
    this.weights['kernel'].tensor.shape = [kernelH * kernelW, inputChannels]
    this.weights['kernel'].createGLTexture({ type: '2d', format: 'float' });
    if (this.useBias) {
      this.weights['bias'].createGLTexture({ type: '2d', format: 'float' });
    }
  }

  /**
   * Method for computing output dimensions and padding, 
   * based on input dimensions, kernel size, and padding mode.
   *
   * @param {number[]} inputShape
   */
  _calcOutputShape(inputShape) {
    if (this.outputShape && this.inputPadding) {
      return;
    }

    const inputHeight = inputShape[0];
    const inputWidth = inputShape[1];
    const inputChannels = inputShape[2];
    const [filter, kernelH, kernelW] = this.kernelShape;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    if (Array.isArray(this.padding)) {
      const outputHeight = (inputHeight - kernelHDilated + this.padding[0] + 
                          this.padding[1] + this.strides[0]) / this.strides[0];
      const outputWidth = (inputWidth - kernelWDilated + this.padding[2] + 
                          this.padding[3] + this.strides[1]) / this.strides[1];
      this.outputShape = [outputHeight, outputWidth, this.weights['kernel'].tensor.shape[1]];
      this.inputPadding = this.padding;
    } else {
      const outputHeight =
      this.padding === 'SAME'
        ? Math.floor((inputHeight + this.strides[0] - 1) / this.strides[0])
        : Math.floor((inputHeight - kernelHDilated + this.strides[0]) / this.strides[0]);
      const outputWidth =
        this.padding === 'SAME'
          ? Math.floor((inputWidth + this.strides[1] - 1) / this.strides[1])
          : Math.floor((inputWidth - kernelWDilated + this.strides[1]) / this.strides[1]);

      const paddingHeight =
        this.padding === 'SAME'
          ? Math.max(0, Math.floor((outputHeight - 1) * this.strides[0] + kernelHDilated - inputHeight))
          : 0;
      const paddingWidth =
        this.padding === 'SAME'
          ? Math.max(0, Math.floor((outputWidth - 1) * this.strides[1] + kernelWDilated - inputWidth))
          : 0;
          
      const paddingHeightBefore = Math.floor(paddingHeight / 2);
      const paddingHeightAfter = paddingHeight - paddingHeightBefore;
      const paddingWidthBefore = Math.floor(paddingWidth / 2);
      const paddingWidthAfter = paddingWidth - paddingWidthBefore;
      this.outputShape = [outputHeight, outputWidth, this.weights['kernel'].tensor.shape[1]];
      this.inputPadding = [paddingHeightBefore, paddingHeightAfter, paddingWidthBefore, paddingWidthAfter];
    }
  }

  /**
   * Pad input tensor if necessary, for padding='SAME'. See above for notes on calculating padding.
   *
   * @param {Tensor} x
   * @param {number} [padValue]
   * @returns {Tensor}
   */
  _padInput(x, padValue = 0) {
    if (this.padding === 'SAME' || Array.isArray(this.padding)) {
      // Test all 0.
      if (this.inputPadding.every((x)=>!x)) {
        return x;
      }
      const [inputHeight, inputWidth, inputChannels] = x.tensor.shape;
      const [paddingHeightBefore, paddingHeightAfter, paddingWidthBefore, paddingWidthAfter] = this.inputPadding;
      const newHeight = inputHeight + paddingHeightBefore + paddingHeightAfter;
      const newWidth = inputWidth + paddingWidthBefore + paddingWidthAfter;
      const _x = new Tensor([], [newHeight, newWidth, inputChannels]);
      if (padValue !== 0) {
        ops.assigns(_x.tensor, padValue);
      }
      ops.assign(
        _x.tensor
          .hi(inputHeight + paddingHeightBefore, inputWidth + paddingWidthBefore, inputChannels)
          .lo(paddingHeightBefore, paddingWidthBefore, 0),
        x.tensor
      );
      return _x;
    }
    return x;
  }

  /**
   * Convert input tensor to column matrix
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  _im2col(x) {
    const [inputHeight, inputWidth, inputChannels] = x.tensor.shape;
    const kernelH = this.kernelShape[1];
    const kernelW = this.kernelShape[2];
    const outputHeight = this.outputShape[0];
    const outputWidth = this.outputShape[1];
    const nbPatches = outputHeight * outputWidth;
    const patchLen = inputChannels * kernelH * kernelW;
    const length = kernelH * kernelW;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    if (!this.imColsMat) {
      this.imColsMat = new Tensor([], [nbPatches, patchLen]);
    }

    const patch = new Tensor([], [kernelH, kernelW, 1]);
    let offset = 0;
    for (let i = 0, limit = inputHeight - kernelHDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputWidth - kernelWDilated; j <= limit; j += this.strides[1]) {
        for (let c = 0; c < inputChannels; ++c) {
          ops.assign(
            patch.tensor,
            x.tensor
              .hi(i + kernelHDilated, j + kernelWDilated, c + 1)
              .lo(i, j, c)
              .step(this.dilationRate[0], this.dilationRate[1], 1)
          );
          this.imColsMat.tensor.data.set(patch.tensor.data, offset);
          offset += length;
        }
      }
    }
    return this.imColsMat;
  }

  /**
   * Pre-compute index map for pooling function
   */
  _createIndexMap() {
    if (this.indexMap) {
      return;
    }

    let [inputHeight, inputWidth, inputChannels] = this.inputShape;
    let rowIndices = new Tensor([], [inputHeight, inputWidth]);
    let index = 0;
    for (let i = 0; i < inputHeight; i++) {
      for (let j = 0; j < inputWidth; j++) {
        rowIndices.tensor.set(i, j, index);
        index += 1;
      }
    }

    // padding
    if (this.padding === 'SAME' || Array.isArray(this.padding)) {
      const [paddingHeightBefore, paddingHeightAfter, paddingWidthBefore, paddingWidthAfter] = this.inputPadding;
      inputHeight = inputHeight + paddingHeightBefore + paddingHeightAfter;
      inputWidth = inputWidth + paddingWidthBefore + paddingWidthAfter;
      const _rowIndices = new Tensor([], [inputHeight, inputWidth]);
      ops.assigns(_rowIndices.tensor, -1);
      ops.assign(
        _rowIndices.tensor
          .hi(this.inputShape[0] + paddingHeightBefore, this.inputShape[1] + paddingWidthBefore)
          .lo(paddingHeightBefore, paddingWidthBefore),
        rowIndices.tensor
      );
      rowIndices.tensor = _rowIndices.tensor;
    }

    const [filters, kernelHeight, kernelWidth] = this.kernelShape;
    const outputHeight = this.outputShape[0];
    const outputWidth = this.outputShape[1];

    this.indexMap = new Tensor([], [outputHeight * outputWidth, kernelHeight * kernelWidth], Int32Array);

    const patchRow = new Tensor([], [kernelHeight, kernelWidth]);
    let offset = 0;
    for (let i = 0, limit = inputHeight - kernelHeight; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputWidth - kernelWidth; j <= limit; j += this.strides[1]) {
        ops.assign(patchRow.tensor, rowIndices.tensor.hi(i + kernelHeight, j + kernelWidth).lo(i, j));
        this.indexMap.tensor.data.set(patchRow.tensor.data, offset);
        offset += kernelHeight * kernelWidth;
      }
    }
    this.indexMap.createGLTexture({ type: '2d', format: 'int', supportSliceTexture: true });
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    const [filter, kernelH, kernelW] = this.kernelShape;
    let outputTextureShape;
    if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
      this._calcOutputShape(this.inputShape);
      this._createIndexMap();
      outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
    } else {
      this.inputShape = x.tensor.shape;
      this._calcOutputShape(this.inputShape);
      x = this._padInput(x);
      this._im2col(x);
      this.imColsMat.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      outputTextureShape = [this.imColsMat.textureShape[0], this.outputShape[2]];
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], outputTextureShape);
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    if (x.is2DReshaped) {
      const hasFragments = Boolean(x.textureSlices);
      if (hasFragments) {
        x.convertTextureSlicesToColStackTexture();
      }
      if (!this.depthwiseConv2DProgram) {
        const depthwiseConv2DShaderSource = depthwiseConv2DShaderSourceFunc(
          this.inputShape[2],
          this.output.textureShape[1],
          this.depthMultiplier,
          this.useBias,
          hasFragments,
          this.fuseActivation
        );
        this.depthwiseConv2DProgram = webgl2.createProgram(depthwiseConv2DShaderSource);
      }
      webgl2.runProgram({
        program: this.depthwiseConv2DProgram,
        output: this.output,
        inputs: [
          { input: x, name: 'x' },
          { input: this.indexMap, name: 'indexMap' },
          { input: this.weights['kernel'], name: 'kernel' },
          ...(this.useBias ? [{ input: this.weights['bias'], name: 'bias' }] : [])
        ],
        supportSliceTexture: true
      });
    } else {
      // run matrix multiply on result of im2col
      const matMulInputs = [{ input: this.imColsMat, name: 'A' }, { input: this.weights['kernel'], name: 'B' }];
      if (this.useBias) {
        matMulInputs.push({ input: this.weights['bias'], name: 'C' });
      }
      if (!this.matMulDepthwiseProgram) {
        this.matMulDepthwiseProgram = webgl2.createProgram(matMulDepthwiseShaderSourceFunc(this.fuseActivation));
      }
      webgl2.runProgram({
        program: this.matMulDepthwiseProgram,
        output: this.output,
        inputs: matMulInputs,
        uniforms: [{ value: this.useBias ? 1 : 0, type: 'bool', name: 'addC' },
                   { value: this.inputShape[2], type: 'int', name: 'inputChannels' },
                   { value: outputTextureShape[1], type: 'int', name: 'outputChannels' },
                   { value: this.depthMultiplier, type: 'int', name: 'depthMultiplier' }],
        supportSliceTexture: true
      });
    }

    // convert back to channels_first ordering if necessary
    if (this.dataFormat === 'NCHW') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(0, 3, 1, 2);
    }
    return this.output;
  }
}
