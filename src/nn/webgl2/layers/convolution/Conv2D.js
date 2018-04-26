import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../webgl/fragmentShader/activation'
import webgl2 from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import ops from 'ndarray-ops'
import matMulShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/matMul'
import conv2dShaderSourceFunc from '../../webgl/fragmentShader/convolution/conv2d'
import { fuseShaderSource } from '../../webgl/fragmentShader/activation/fuse'

/**
 * Conv2D layer class
 */
export default class Conv2D extends Layer {
  /**
   * Creates a Conv2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number} [attrs.filters] - Number of convolution filters to use
   * @param {number|number[]} [attrs.kernel_size] - Size of the convolution kernel
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'Conv2D';

    const {
      filters = 1,
      kernel_size = [3, 3],
      strides = [1, 1],
      padding = 'VALID',
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

    this.pointwise = false;
    if (this.kernelShape[1] === 1 && this.kernelShape[2] === 1 && 
        this.strides[0] === 1 && this.strides[1] === 1 &&
        (padding === 'VALID' || padding === 'SAME')) {
      this.pointwise = true;
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
    const params = this.useBias ? ['kernel', 'bias'] : ['kernel'];

    if (this.dataFormat === 'NCHW') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(0, 2, 3, 1);
    }
    
    super.setWeights(params, weightsArr, false);
    this._w2row();
    this.weights['kernel'] = this.wRowsMat;
    this.weights['kernel'].createGLTexture({ type: '2d', format: 'float' });
    // console.log(`webgl2.MAX_TEXTURE_SIZE: ${webgl2.MAX_TEXTURE_SIZE}`)
    // console.log(this.weights['kernel'])
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
    const [filter, kernelH, kernelW] = this.kernelShape;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    if (Array.isArray(this.padding)) {
      const outputHeight = (inputHeight - kernelHDilated + this.padding[0] + 
                          this.padding[1] + this.strides[0]) / this.strides[0];
      const outputWidth = (inputWidth - kernelWDilated + this.padding[2] + 
                          this.padding[3] + this.strides[1]) / this.strides[1];
      this.outputShape = [outputHeight, outputWidth, filter];
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
      this.outputShape = [outputHeight, outputWidth, filter];
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
    const patchLen = kernelH * kernelW * inputChannels;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    if (!this.imColsMat) {
      this.imColsMat = new Tensor([], [nbPatches, patchLen]);
    }

    // if Pointwise Convolution
    if (kernelHDilated === 1 && kernelWDilated === 1 && this.strides[0] === 1 && this.strides[1] === 1) {
      this.imColsMat.replaceTensorData(x.tensor.data);
      return this.imColsMat;
    }

    const patch = new Tensor([], [kernelH, kernelW, inputChannels]);
    let offset = 0;
    for (let i = 0, limit = inputHeight - kernelHDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputWidth - kernelWDilated; j <= limit; j += this.strides[1]) {
        ops.assign(
          patch.tensor,
          x.tensor
            .hi(i + kernelHDilated, j + kernelWDilated, inputChannels)
            .lo(i, j, 0)
            .step(this.dilationRate[0], this.dilationRate[1], 1)
        );
        this.imColsMat.tensor.data.set(patch.tensor.data, offset);
        offset += patchLen;
      }
    }

    return this.imColsMat;
  }

  /**
   * Convert filter weights to row matrix
   *
   * @returns {Tensor}
   */
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[3];
    const [filter, kernelH, kernelW] = this.kernelShape;
    const patchLen = kernelH * kernelW * inputChannels;

    this.wRowsMat = new Tensor([], [patchLen, filter]);

    const patch = new Tensor([], [kernelH, kernelW, inputChannels]);
    const patchRaveled = new Tensor([], [patchLen]);
    for (let n = 0; n < filter; n++) {
      ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(n, null, null, null));
      patchRaveled.replaceTensorData(patch.tensor.data);
      ops.assign(this.wRowsMat.tensor.pick(null, n), patchRaveled.tensor);
    }

    return this.wRowsMat;
  }

  /**
   * Creates a index mapping from the 2D-reshaped input tensor with associated 3D tensor shape to the representation
   * required prior to the matrix multiply. This allows us to work directly on the 2D tensor representations rather
   * than needing to reshape to the 3D reprentation and calling im2col.
   *
   * @param {Object} indicesForReshaped
   */
  _createIndexMap(indicesForReshaped) {
    if (this.indexMap) {
      return;
    }

    let [inputHeight, inputWidth, inputChannels] = this.inputShape;

    let indices = new Tensor(indicesForReshaped.data, indicesForReshaped.shape, Int32Array);

    // padding
    if (this.padding === 'SAME' || Array.isArray(this.padding)) {
      const [paddingHeightBefore, paddingHeightAfter, paddingWidthBefore, paddingWidthAfter] = this.inputPadding;
      inputHeight = inputHeight + paddingHeightBefore + paddingHeightAfter;
      inputWidth = inputWidth + paddingWidthBefore + paddingWidthAfter;
      const padValue = -1;
      indices = this._padInput(indices, padValue);
    }

    const kernelH = this.kernelShape[1];
    const kernelW = this.kernelShape[2];
    const outputHeight = this.outputShape[0];
    const outputWidth = this.outputShape[1];
    const nbPatches = outputHeight * outputWidth;
    const patchLen = kernelH * kernelW * inputChannels;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);
    // console.log(nbPatches, patchLen);
    this.indexMap = new Tensor([], [nbPatches, patchLen], Int32Array);

    // if Pointwise Convolution
    if (kernelHDilated === 1 && kernelWDilated === 1 && this.strides[0] === 1 && this.strides[1] === 1) {
      this.indexMap.replaceTensorData(indices.tensor.data);
    } else {
      const indicesPatch = new Tensor([], [kernelH, kernelW, inputChannels]);
      let offset = 0
      for (let i = 0, limit = inputHeight - kernelHDilated; i <= limit; i += this.strides[0]) {
        for (let j = 0, limit = inputWidth - kernelWDilated; j <= limit; j += this.strides[1]) {
          ops.assign(
            indicesPatch.tensor,
            indices.tensor
              .hi(i + kernelHDilated, j + kernelWDilated, inputChannels)
              .lo(i, j, 0)
              .step(this.dilationRate[0], this.dilationRate[1], 1)
          );
          this.indexMap.tensor.data.set(indicesPatch.tensor.data, offset);
          offset += patchLen;
        }
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
    let outputTextureShape;
    if (this.pointwise) {
      // console.log('[Conv2D] use pointwise!');
      this.inputShape = x.originalShape;
      this.outputShape= [x.originalShape[0], x.originalShape[1], this.kernelShape[0]];
      outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
    } else if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
      this._calcOutputShape(this.inputShape);
      this._createIndexMap(x.indicesForReshaped);
      outputTextureShape = [this.indexMap.textureShape[0], this.weights['kernel'].textureShape[1]];
    } else {
      // console.log('[Conv2D] x is not 2DReshaped!');
      this.inputShape = x.tensor.shape;
      this._calcOutputShape(this.inputShape);
      x = this._padInput(x);
      this._im2col(x);
      if (!this.imColsMat.textureShape) {
        this.imColsMat.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      } else {
        this.imColsMat.replaceTensorData(this.imColsMat.tensor.data);
      }
      outputTextureShape = [this.imColsMat.textureShape[0], this.weights['kernel'].textureShape[1]];
    }

    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], outputTextureShape);
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    if (this.pointwise) {
      // run 1x1 pointwise conv
      const matMulInputs = [{ input: x, name: 'A' }, { input: this.weights['kernel'], name: 'B' }];
      if (this.useBias) {
        matMulInputs.push({ input: this.weights['bias'], name: 'C' });
      }
      if (!this.matMulProgram) {
        this.matMulProgram = webgl2.createProgram(matMulShaderSourceFunc(this.fuseActivation));
      }
      webgl2.runProgram({
        program: this.matMulProgram,
        output: this.output,
        inputs: matMulInputs,
        uniforms: [{ value: this.useBias ? 1 : 0, type: 'bool', name: 'addC' }],
        supportSliceTexture: true
      });
    } else if (x.is2DReshaped) {
      // run conv2d program, which involves mapping the input using indexMap, and matrix multiply with weights
      const hasFragments = Boolean(x.textureSlices);
      if (hasFragments) {
        x.convertTextureSlicesToColStackTexture();
      }
      if (!this.conv2dProgram) {
        const conv2dProgramSource = conv2dShaderSourceFunc(
          this.output.textureSliceShape ? this.output.textureSliceShape : this.output.textureShape,
          x.textureSliceShape ? x.textureSliceShape : x.textureShape,
          this.indexMap.textureSliceShape ? this.indexMap.textureSliceShape : this.indexMap.textureShape,
          this.useBias,
          hasFragments,
          this.fuseActivation
        );
        this.conv2dProgram = webgl2.createProgram(conv2dProgramSource);
      }
      webgl2.runProgram({
        program: this.conv2dProgram,
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
      if (!this.matMulProgram) {
        this.matMulProgram = webgl2.createProgram(matMulShaderSourceFunc(this.fuseActivation));
      }
      webgl2.runProgram({
        program: this.matMulProgram,
        output: this.output,
        inputs: matMulInputs,
        uniforms: [{ value: this.useBias ? 1 : 0, type: 'bool', name: 'addC' }],
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
