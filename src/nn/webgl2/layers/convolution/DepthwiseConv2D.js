import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../webgl/fragmentShader/activation'
import webgl2 from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import ops from 'ndarray-ops'
import { matMulDepthwiseShaderSource } from '../../webgl/fragmentShader/arithmetic/matMulDepthwise'
import DepthwiseConv2DShaderSourceFunc from '../../webgl/fragmentShader/convolution/DepthwiseConv2D'
import DepthwiseConv2DColStackShaderSourceFunc from '../../webgl/fragmentShader/convolution/DepthwiseConv2DColStack'

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

    if (padding === 'VALID' || padding === 'SAME') {
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

    this.activation = activation;
    if (this.activation !== 'NONE') {
      this.activationProgram = webgl2.createProgram(activations[this.activation]);
    }

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
    const inputChannels = weightsArr[0].tensor.shape[3];
    const [filter, kernelH, kernelW] = this.kernelShape;
    
    if (this.dataFormat === 'NCHW') {
      weightsArr[0].tensor = weightsArr[0].tensor.transpose(0, 2, 3, 1);
    }

    super.setWeights(params, weightsArr, false);
    if (inputChannels * kernelH * kernelW <= webgl2.MAX_TEXTURE_SIZE) {
      this._w2row();
    } else {
      this._w2rowColStack();
    }
    
    this.weights['kernel'] = this.wRowsMat;
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

    const inputRows = inputShape[0];
    const inputCols = inputShape[1];
    const [filter, kernelH, kernelW] = this.kernelShape;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    const outputRows =
      this.padding === 'SAME'
        ? Math.floor((inputRows + this.strides[0] - 1) / this.strides[0])
        : Math.floor((inputRows - kernelHDilated + this.strides[0]) / this.strides[0]);
    const outputCols =
      this.padding === 'SAME'
        ? Math.floor((inputCols + this.strides[1] - 1) / this.strides[1])
        : Math.floor((inputCols - kernelWDilated + this.strides[1]) / this.strides[1]);
    const outputChannels = inputShape[2];

    const paddingRow =
      this.padding === 'SAME'
        ? Math.max(0, Math.floor((outputRows - 1) * this.strides[0] + kernelHDilated - inputRows))
        : 0;
    const paddingCol =
      this.padding === 'SAME'
        ? Math.max(0, Math.floor((outputCols - 1) * this.strides[1] + kernelWDilated - inputCols))
        : 0;
    const paddingRowBefore = Math.ceil(paddingRow / 2);
    const paddingRowAfter = paddingRow - paddingRowBefore;
    const paddingColBefore = Math.ceil(paddingCol / 2);
    const paddingColAfter = paddingCol - paddingColBefore;
    this.outputShape = [outputRows, outputCols, outputChannels];
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter];
  }

  /**
   * Pad input tensor if necessary, for padding='SAME'. See above for notes on calculating padding.
   *
   * @param {Tensor} x
   * @param {number} [padValue]
   * @returns {Tensor}
   */
  _padInput(x, padValue = 0) {
    if (this.padding === 'SAME') {
      // Test all 0.
      let flag = false;
      this.inputPadding.forEach(pad => {
        if (pad) {
          flag = true;
        }
      });
      if (!flag) {
        return x;
      }
      const [inputRows, inputCols, inputChannels] = x.tensor.shape;
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      const newRows = inputRows + paddingRowBefore + paddingRowAfter;
      const newCols = inputCols + paddingColBefore + paddingColAfter;
      const _x = new Tensor([], [newRows, newCols, inputChannels]);
      if (padValue !== 0) {
        ops.assigns(_x.tensor, padValue);
      }
      ops.assign(
        _x.tensor
          .hi(inputRows + paddingRowBefore, inputCols + paddingColBefore, inputChannels)
          .lo(paddingRowBefore, paddingColBefore, 0),
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
    const [inputRows, inputCols, inputChannels] = x.tensor.shape;
    const kernelH = this.kernelShape[1];
    const kernelW = this.kernelShape[2];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols;
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
    for (let i = 0, limit = inputRows - kernelHDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - kernelWDilated; j <= limit; j += this.strides[1]) {
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
   * Convert filter weights to row matrix
   *
   * @returns {Tensor}
   */
  _w2row() {
    const inputChannels = this.weights['kernel'].tensor.shape[3];
    const [filter, kernelH, kernelW] = this.kernelShape;
    const patchLen = kernelH * kernelW * inputChannels;
    const length = kernelH * kernelW;

    this.wRowsMat = new Tensor([], [filter, patchLen]);

    const patch = new Tensor([], [kernelH, kernelW]);
    const patchRaveled = new Tensor([], [patchLen]);
    let offset = 0;
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(0, null, null, c));
      this.wRowsMat.tensor.data.set(patch.tensor.data, offset);
      offset += length;
    }
    return this.wRowsMat;
  }
  
  _w2rowColStack() {
    const inputChannels = this.weights['kernel'].tensor.shape[3];
    const [filter, kernelH, kernelW] = this.kernelShape;
    const patchLen = kernelH * kernelW;
    const length = kernelH * kernelW;

    this.wRowsMat = new Tensor([], [filter * inputChannels, patchLen]);

    const patch = new Tensor([], [kernelH, kernelW]);
    const patchRaveled = new Tensor([], [patchLen]);
    let offset = 0;
    for (let c = 0; c < inputChannels; c++) {
      ops.assign(patch.tensor, this.weights['kernel'].tensor.pick(0, null, null, c));
      this.wRowsMat.tensor.data.set(patch.tensor.data, offset);
      offset += length;
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

    let [inputRows, inputCols, inputChannels] = this.inputShape;

    let indices = new Tensor(indicesForReshaped.data, indicesForReshaped.shape, Int32Array);

    // padding for border mode 'SAME'
    if (this.padding === 'SAME') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      inputRows = inputRows + paddingRowBefore + paddingRowAfter;
      inputCols = inputCols + paddingColBefore + paddingColAfter;
      const padValue = -1;
      indices = this._padInput(indices, padValue);
    }

    const kernelH = this.kernelShape[1];
    const kernelW = this.kernelShape[2];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols;
    const patchLen = inputChannels * kernelH * kernelW;
    const length = kernelH * kernelW;;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    this.indexMap = new Tensor([], [nbPatches, patchLen], Int32Array);

    const indicesPatch = new Tensor([], [kernelH, kernelW, 1]);
    let offset = 0;
    for (let i = 0, limit = inputRows - kernelHDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - kernelWDilated; j <= limit; j += this.strides[1]) {
        for (let c = 0; c < inputChannels; ++c) {
          ops.assign(
            indicesPatch.tensor,
            indices.tensor
              .hi(i + kernelHDilated, j + kernelWDilated, c + 1)
              .lo(i, j, c)
              .step(this.dilationRate[0], this.dilationRate[1], 1)
          );
          this.indexMap.tensor.data.set(indicesPatch.tensor.data, offset);
          offset += length;
        }
      }
    }
    this.indexMap.createGLTexture({ type: '2d', format: 'int', supportSliceTexture: true });
  }

  /**
   * Creates a index mapping from the 2D-reshaped input tensor with associated 3D tensor shape to the representation
   * required prior to the matrix multiply. This allows us to work directly on the 2D tensor representations rather
   * than needing to reshape to the 3D reprentation and calling im2col.
   *
   * @param {Object} indicesForReshaped
   */
  _createIndexMapColStack(indicesForReshaped) {
    if (this.indexMap) {
      return;
    }

    let [inputRows, inputCols, inputChannels] = this.inputShape;

    let indices = new Tensor(indicesForReshaped.data, indicesForReshaped.shape, Int32Array);

    // padding for border mode 'SAME'
    if (this.padding === 'SAME') {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      inputRows = inputRows + paddingRowBefore + paddingRowAfter;
      inputCols = inputCols + paddingColBefore + paddingColAfter;
      const padValue = -1;
      indices = this._padInput(indices, padValue);
    }

    const kernelH = this.kernelShape[1];
    const kernelW = this.kernelShape[2];
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];
    const nbPatches = outputRows * outputCols * kernelH * kernelW;
    const patchLen = inputChannels ;
    const length = kernelH * kernelW;

    // effective shape after filter dilation
    const kernelHDilated = kernelH + (kernelH - 1) * (this.dilationRate[0] - 1);
    const kernelWDilated = kernelW + (kernelW - 1) * (this.dilationRate[1] - 1);

    this.indexMap = new Tensor([], [nbPatches, patchLen], Int32Array);
    const indicesPatch = new Tensor([], [1, 1, inputChannels]);
    let offset = 0;
    let cnt = 0;
    for (let i = 0, limit = inputRows - kernelHDilated; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - kernelWDilated; j <= limit; j += this.strides[1]) {
        for (let m = 0; m < kernelHDilated; ++m) {
          for (let n = 0; n < kernelWDilated; ++n) {
            ops.assign(
              indicesPatch.tensor,
              indices.tensor
                .hi(i + m + 1, j + n + 1, inputChannels)
                .lo(i + m, j + n, 0)
            );
            this.indexMap.tensor.data.set(indicesPatch.tensor.data, offset);
            offset += inputChannels;

          }
        }
        ++cnt;
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
    if (x.is2DReshaped || x.is2DSquareReshaped) {
      if (x.tensor.shape[1] * kernelH * kernelW <= webgl2.MAX_TEXTURE_SIZE) {
        this.inputShape = x.originalShape;
        this._calcOutputShape(this.inputShape);
        this._createIndexMap(x.indicesForReshaped);
        outputTextureShape = [this.indexMap.textureShape[0], this.outputShape[2]];
      } else {
        this.inputShape = x.originalShape;
        this._calcOutputShape(this.inputShape);
        this._createIndexMapColStack(x.indicesForReshaped);
        outputTextureShape = [this.outputShape[0] * this.outputShape[1], this.outputShape[2]];
      }
    } else {
      this.inputShape = x.tensor.shape;
      this._calcOutputShape(this.inputShape);
      x = this._padInput(x);
      this._im2col(x);
      this.imColsMat.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      outputTextureShape = [this.imColsMat.textureShape[0], this.outputShape[2]];
    }

    // create output textures if doesn't already exist
    if (this.activation !== 'NONE' && !this.outputPreactiv) {
      this.outputPreactiv = new Tensor([], outputTextureShape);
      this.outputPreactiv.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      this.outputPreactiv.is2DReshaped = true;
      this.outputPreactiv.originalShape = this.outputShape;
      this.outputPreactiv.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }
    if (!this.output) {
      this.output = new Tensor([], outputTextureShape);
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      this.output.is2DReshaped = true;
      this.output.originalShape = this.outputShape;
      this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.outputShape, false, -1);
    }

    if (x.is2DReshaped || x.is2DSquareReshaped) {
      if (x.tensor.shape[1] * kernelH * kernelW <= webgl2.MAX_TEXTURE_SIZE) {
        // run conv2d program, which involves mapping the input using indexMap, and matrix multiply with weights
        const hasFragments = Boolean(x.textureSlices);
        if (hasFragments) {
          x.convertTextureSlicesToColStackTexture();
        }
        if (!this.DepthwiseConv2DProgram) {
          const DepthwiseConv2DShaderSource = DepthwiseConv2DShaderSourceFunc(
            this.output.textureSliceShape ? this.output.textureSliceShape : this.output.textureShape,
            x.textureSliceShape ? x.textureSliceShape : x.textureShape,
            this.indexMap.textureSliceShape ? this.indexMap.textureSliceShape : this.indexMap.textureShape,
            this.useBias,
            hasFragments
          );
          this.DepthwiseConv2DProgram = webgl2.createProgram(DepthwiseConv2DShaderSource);
        }
        webgl2.runProgram({
          program: this.DepthwiseConv2DProgram,
          output: this.activation === 'NONE' ? this.output : this.outputPreactiv,
          inputs: [
            { input: x, name: 'x' },
            { input: this.indexMap, name: 'indexMap' },
            { input: this.weights['kernel'], name: 'kernel' },
            ...(this.useBias ? [{ input: this.weights['bias'], name: 'bias' }] : [])
          ],
          supportSliceTexture: true
        });
        // if (hasFragments) {
          // x.deleteColStackTexture();
        // }
      } else {
        if (this.output.textureSliceShape) {
          throw new Error('Texture is not large enough.');
        }
        const hasFragments = Boolean(x.textureSlices);
        if (hasFragments) {
          x.convertTextureSlicesToColStackTexture();
        }

        if (!this.DepthwiseConv2DProgram) {
          const DepthwiseConv2DColStackShaderSource = DepthwiseConv2DColStackShaderSourceFunc(
            this.output.textureShape,
            x.textureSliceShape ? x.textureSliceShape : x.textureShape,
            this.indexMap.textureShape,
            this.useBias,
            hasFragments
          );
          this.DepthwiseConv2DColStackProgram = webgl2.createProgram(DepthwiseConv2DColStackShaderSource);
        }
        webgl2.runProgram({
          program: this.DepthwiseConv2DColStackProgram,
          output: this.activation === 'NONE' ? this.output : this.outputPreactiv,
          inputs: [
            { input: x, name: 'x' },
            { input: this.indexMap, name: 'indexMap' },
            { input: this.weights['kernel'], name: 'kernel' },
            ...(this.useBias ? [{ input: this.weights['bias'], name: 'bias' }] : [])
          ],
          supportSliceTexture: true
        });
        // if (hasFragments) {
        //   x.deleteColStackTexture()
        // }
      }

    } else {
      // run matrix multiply on result of im2col
      const matMulInputs = [{ input: this.imColsMat, name: 'A' }, { input: this.weights['kernel'], name: 'B' }];
      if (this.useBias) {
        matMulInputs.push({ input: this.weights['bias'], name: 'C' });
      }
      if (!this.matMulDepthwiseProgram) {
        this.matMulDepthwiseProgram = webgl2.createProgram(matMulDepthwiseShaderSource);
      }
      webgl2.runProgram({
        program: this.matMulDepthwiseProgram,
        output: this.activation === 'NONE' ? this.output : this.outputPreactiv,
        inputs: matMulInputs,
        uniforms: [{ value: this.useBias ? 1 : 0, type: 'bool', name: 'addC' },
                   { value: outputTextureShape[1], type: 'int', name: 'inputChannels' }],
        supportSliceTexture: true
      });
    }

    // Activation
    if (this.activation !== 'NONE') {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.output,
        inputs: [{ input: this.outputPreactiv, name: 'x' }],
        supportSliceTexture: true
      });
    }
    // this.output.transferFromGLTexture()

    //   // convert back to channels_first ordering if necessary
    //   if (this.dataFormat === 'NCHW') {
    //     weightsArr[0].tensor = weightsArr[0].tensor.transpose(0, 3, 1, 2);
    //   }
    // }
    return this.output;
  }
}
