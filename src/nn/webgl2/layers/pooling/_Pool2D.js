import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../webgl/fragmentShader/activation'
import webgl2 from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import ops from 'ndarray-ops'
import pool2DShaderSourceFunc from '../../webgl/fragmentShader/pooling/pool2D'

/**
 * _Pool2D layer class
 */
export default class _Pool2D extends Layer {
  /**
   * Creates a _Pool2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = '_Pool2D';
    const { 
      kernel_size = [2, 2], 
      strides = [2, 2], 
      padding = 'VALID', 
      data_format = 'NHWC',
      activation = 'NONE'
    } = attrs;

    if (Array.isArray(kernel_size)) {
      this.kernelShape = kernel_size;
    } else {
      this.kernelShape = [kernel_size, kernel_size];
    }

    if (Array.isArray(strides)) {
      this.strides = strides;
    } else {
      this.strides = [strides, strides];
    }

    if (Array.isArray(padding)) {
      if (padding.length !== 4) {
        this.throwError('Invalid padding.');
        // if all numbers in padding are 0, use padding = 'VALID'
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

    if (data_format === 'NHWC') {
      this.dataFormat = data_format;
    } else {
      this.throwError('Only NHWC data formats are allowed.');
    }

    this.activation = activation;
    if (this.activation !== 'NONE') {
      this.activationProgram = webgl2.createProgram(activations[this.activation]);
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

    const [inputRows, inputCols, inputChannels] = inputShape
    const [kernelH, kernelW] = this.kernelShape;

    if (Array.isArray(this.padding)) {
      const outputRows = Math.floor((inputRows - kernelH + this.strides[0] + this.padding[0]+this.padding[1]) / this.strides[0]);
      const outputCols = Math.floor((inputCols - kernelW + this.strides[1] + this.padding[2]+this.padding[3]) / this.strides[1]);
      this.outputShape = [outputRows, outputCols, inputChannels];
      this.inputPadding = this.padding;
    } else {
      const outputRows =
      this.padding === 'SAME'
        ? Math.floor((inputRows + this.strides[0] - 1) / this.strides[0])
        : Math.floor((inputRows - kernelH + this.strides[0]) / this.strides[0]);
      const outputCols =
        this.padding === 'SAME'
          ? Math.floor((inputCols + this.strides[1] - 1) / this.strides[1])
          : Math.floor((inputCols - kernelW + this.strides[1]) / this.strides[1]);

      const paddingRow =
        this.padding === 'SAME'
          ? Math.max(0, Math.floor((outputRows - 1) * this.strides[0] + kernelH - inputRows))
          : 0;
      const paddingCol =
        this.padding === 'SAME'
          ? Math.max(0, Math.floor((outputCols - 1) * this.strides[1] + kernelW - inputCols))
          : 0;
    const paddingRowBefore = Math.floor(paddingRow / 2);
    const paddingRowAfter = paddingRow - paddingRowBefore;
    const paddingColBefore = Math.floor(paddingCol / 2);
    const paddingColAfter = paddingCol - paddingColBefore;
    this.outputShape = [outputRows, outputCols, inputChannels];
    this.inputPadding = [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter];
    }
  }

  /**
   * Pre-compute index map for pooling function
   */
  _createIndexMap() {
    if (this.poolIndexMap) {
      return;
    }

    let [inputRows, inputCols, inputChannels] = this.inputShape;
    const rowIndices = new Tensor([], [inputRows, inputCols]);
    let index = 0;
    for (let i = 0; i < inputRows; i++) {
      for (let j = 0; j < inputCols; j++) {
        rowIndices.tensor.set(i, j, index);
        index += 1;
      }
    }

    // padding
    if (this.padding === 'SAME' || Array.isArray(this.padding)) {
      const [paddingRowBefore, paddingRowAfter, paddingColBefore, paddingColAfter] = this.inputPadding;
      inputRows = inputRows + paddingRowBefore + paddingRowAfter;
      inputCols = inputCols + paddingColBefore + paddingColAfter;
      const _rowIndices = new Tensor([], [inputRows, inputCols]);
      ops.assigns(_rowIndices.tensor, -1);
      ops.assign(
        _rowIndices.tensor
          .hi(this.inputShape[0] + paddingRowBefore, this.inputShape[1] + paddingColBefore)
          .lo(paddingRowBefore, paddingColBefore),
        rowIndices.tensor
      );
      rowIndices.tensor = _rowIndices.tensor;
    }

    const [nbRow, nbCol] = this.kernelShape;
    const outputRows = this.outputShape[0];
    const outputCols = this.outputShape[1];

    this.poolIndexMap = new Tensor([], [outputRows * outputCols, nbRow * nbCol], Int32Array);

    const patchRow = new Tensor([], [nbRow, nbCol]);
    let offset = 0;
    for (let i = 0, limit = inputRows - nbRow; i <= limit; i += this.strides[0]) {
      for (let j = 0, limit = inputCols - nbCol; j <= limit; j += this.strides[1]) {
        ops.assign(patchRow.tensor, rowIndices.tensor.hi(i + nbRow, j + nbCol).lo(i, j));
        this.poolIndexMap.tensor.data.set(patchRow.tensor.data, offset);
        offset += nbRow * nbCol;
      }
    }
    this.poolIndexMap.createGLTexture({ type: '2d', format: 'int', supportSliceTexture: true });
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
      this._calcOutputShape(this.inputShape);
      this._createIndexMap();
    } else {
      this.throwError('Invalid input.');
    }
    const [outputRows, outputCols, inputChannels] = this.outputShape;
    const outputTextureShape = [outputRows * outputCols, inputChannels];
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

    const poolSize = this.kernelShape[0] * this.kernelShape[1];
    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max';
    const programUniforms = [
      { value: this.output.textureShape[1], type: 'int', name: 'channels' },
      { value: poolSize, type: 'int', name: 'poolSize' },
      { value: +isMaxPooling, type: 'bool', name: 'isMaxPooling' }
    ];
    if (x.textureSlices) {
      x.convertTextureSlicesToColStackTexture();
    }

    if (!this.poolingProgram) {
      const pool2DShaderSource = pool2DShaderSourceFunc(x.textureSlices);
      this.poolingProgram = webgl2.createProgram(pool2DShaderSource);
    }
    webgl2.runProgram({
      program: this.poolingProgram,
      output: this.activation === 'NONE' ? this.output : this.outputPreactiv,
      inputs: [{ input: x, name: 'x' }, { input: this.poolIndexMap, name: 'poolIndexMap' }],
      uniforms: programUniforms,
      supportSliceTexture: true
    });

    // Activation
    if (this.activation !== 'NONE') {
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.output,
        inputs: [{ input: this.outputPreactiv, name: 'x' }],
        supportSliceTexture: true
      });
    }
    return this.output;
  }
}
