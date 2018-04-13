import webgl2 from './WebGL2'
import * as tensorUtils from './utils/tensorUtils'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import squeeze from 'ndarray-squeeze'

/**
 * Tensor class
 */
export default class Tensor {
  /**
   * Creates a tensor
   *
   * @param {(TypedArray|Array)} data
   * @param {number[]} shape
   * @param {Object} [options]
   */
  constructor(data, shape, type) {
    this.is1D = false;
    this.arrayType = type || Float32Array;
    if (data && data.length && (data instanceof this.arrayType || data instanceof Array)) {
      if (data.length && shape.length && data.length !== shape.reduce((a, b) => a * b, 1)) {
        throw new Error('Tensor shape does not match the length of data.');
      }
      if (data instanceof this.arrayType) {
        this.tensor = ndarray(data, shape);
      } else if (data instanceof Array) {
        this.tensor = ndarray(new this.arrayType(data), shape);
      }
    } else if (!data.length && shape.length) {
      // Initialize tensor data with 0 when data is not given
      this.tensor = ndarray(new this.arrayType(shape.reduce((a, b) => a * b, 1)), shape);
    } else {
      this.tensor = ndarray(new this.arrayType([]), []);
    }
  }

  /**
   * Creates WebGL2 texture
   *
   * Without args, defaults to gl.TEXTURE_2D and gl.R32F
   *
   * @param {string} [opts.type]
   * @param {string} [opts.format]
   * @param {boolean} [opts.supportSliceTexture]
   */
  createGLTexture({ type = '2d', format = 'float', supportSliceTexture = false }) {
    let shape = []
    if (this.tensor.shape.length === 1) {
      shape = [1, this.tensor.shape[0]];
      this.is1D = true;
    } else if (this.tensor.shape.length === 2) {
      shape = this.tensor.shape;
    } else {
      throw new Error('Can not create WebGL2 texture for shape length > 2.');
    }

    this.textureShape = shape;
    this.textureType = type;
    this.textureFormat = format;

    if (type === '2d') {
      if (this.textureShape[0] > webgl2.MAX_TEXTURE_SIZE && supportSliceTexture) {
        this._createTextureSlices();
      } else {
        this._create2DGLTexture();
      }
    } else {
      throw new Error(`Invalid type: ${type}`);
    }
  }

  /**
   * Create 2D WebGL2 texture
   * 
   * WebGL1 can only use non power of 2 textures with filtering set to NEAREST or LINEAR and
   * it can not generate a mipmap for them.
   */
  _create2DGLTexture() {
    const gl = webgl2.context;
    const textureOptions = webgl2.getTextureOptions(this.textureType, this.textureFormat);
    const { textureTarget, textureInternalFormat, textureFormat, textureType } = textureOptions;

    this.texture = gl.createTexture();
    webgl2.toDelete.textures.push(this.texture);
    gl.bindTexture(textureTarget, this.texture);

    const shape = this.textureShape;
    const data = this.tensor.data;
    gl.texImage2D(textureTarget, 0, textureInternalFormat, shape[1], shape[0], 0, textureFormat, textureType, data);

    // clamp to edge
    gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    // no interpolation
    gl.texParameteri(textureTarget, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(textureTarget, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  }

  /**
   * For 2D WebGL2 texture with first dimension exceeding webgl2.MAX_TEXTURE_SIZE, slice it and create an array of 2D textures
   */
  _createTextureSlices() {
    const gl = webgl2.context;
    const textureOptions = webgl2.getTextureOptions(this.textureType, this.textureFormat);
    const { textureTarget, textureInternalFormat, textureFormat, textureType } = textureOptions;

    this.textureSlices = [];
    this.textureSliceShape = [webgl2.MAX_TEXTURE_SIZE, this.textureShape[1]];

    const shape = this.textureSliceShape;
    const numSlices = Math.ceil(this.textureShape[0] / webgl2.MAX_TEXTURE_SIZE);
    let offset = 0;

    for (let k = 0; k < numSlices; k++) {
      const texture = gl.createTexture();
      webgl2.toDelete.textures.push(texture);
      gl.bindTexture(textureTarget, texture);

      // append 0s to last fragment
      let data;
      if (k === numSlices - 1) {
        data = new this.arrayType(shape[0] * shape[1]);
        data.set(this.tensor.data.slice(offset, offset + shape[0] * shape[1]), 0);
      } else {
        data = this.tensor.data.slice(offset, offset + shape[0] * shape[1]);
      }
      gl.texImage2D(textureTarget, 0, textureInternalFormat, shape[1], shape[0], 0, textureFormat, textureType, data);

      // clamp to edge
      gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      // no interpolation
      gl.texParameteri(textureTarget, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(textureTarget, gl.TEXTURE_MIN_FILTER, gl.NEAREST);

      this.textureSlices.push(texture);
      offset += shape[0] * shape[1];
    }
    // console.log(`Create Fragments: ${numSlices}`);
  }

  /**
   * Converts an array of horizontal-slice textureSlices into a single column-stacked texture
   */
  convertTextureSlicesToColStackTexture() {
    if (!this.textureSlices || !this.textureSliceShape) {
      throw new Error('No textureSlices available.');
    }

    const gl = webgl2.context;
    const textureOptions = webgl2.getTextureOptions(this.textureType, this.textureFormat)
    const { textureTarget, textureInternalFormat, textureFormat, textureType } = textureOptions

    if (!this.colStackTexture) {
      this.colStackTexture = gl.createTexture()
      webgl2.toDelete.textures.push(this.colStackTexture);
      gl.bindTexture(textureTarget, this.colStackTexture);

      const numSlices = this.textureSlices.length;
      this.colStackTextureShape = [
        this.textureSliceShape[0],
        this.textureSliceShape[1] * numSlices
      ];

      const shape = this.colStackTextureShape;
      const data = new this.arrayType(shape[0] * shape[1]);
      gl.texImage2D(textureTarget, 0, textureInternalFormat, shape[1], shape[0], 0, textureFormat, textureType, data);

      // clamp to edge
      gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(textureTarget, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      // no interpolation
      gl.texParameteri(textureTarget, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(textureTarget, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    } else {
      gl.bindTexture(textureTarget, this.colStackTexture);
    }

    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
    this.textureSlices.forEach((texture, k) => {
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
      gl.copyTexSubImage2D(
        textureTarget,
        0,
        k * this.textureSliceShape[1],
        0,
        0,
        0,
        this.textureSliceShape[1],
        this.textureSliceShape[0]
      )
    });
    gl.deleteFramebuffer(framebuffer);
  }

  /**
   * Removes colStackTexture
   */
  deleteColStackTexture() {
    if (this.colStackTexture) {
      const gl = webgl2.context;
      gl.deleteTexture(this.colStackTexture);
      delete this.colStackTexture;
      delete this.colStackTextureShape;
    }
  }

  /**
   * Deletes WebGLTexture
   */
  deleteGLTexture() {
    webgl2.toDelete.textures.forEach(texture => gl.deleteTexture(texture));
    webgl2.toDelete.textures = [];
  }

  /**
   * Replaces data in the underlying ndarray, and the corresponding WebGLTexture if texture is present
   *
   * @param {number[]} data
   */
  replaceTensorData(data) {
    if (data && data.length && data instanceof this.arrayType) {
      this.tensor.data.set(data);
    } else if (data && data.length && data instanceof Array) {
      this.tensor.data.set(new this.arrayType(data));
    } else {
      throw new Error('Invalid input for replaceTensorData method.');
    }

    if (this.texture) {
      // console.log('inputTensor is texture');
      const gl = webgl2.context;
      const shape = this.textureShape;
      const textureOptions = webgl2.getTextureOptions(this.textureType, this.textureFormat);
      const { textureTarget, textureFormat, textureType } = textureOptions;
      
      gl.bindTexture(textureTarget, this.texture)
      gl.texSubImage2D(textureTarget, 0, 0, 0, shape[1], shape[0], textureFormat, textureType, data, 0)
    } else if (this.textureSlices){
      // console.log('inputTensor is textureSlices');
      const gl = webgl2.context;
      const textureOptions = webgl2.getTextureOptions(this.textureType, this.textureFormat);
      const { textureTarget, textureFormat, textureType } = textureOptions;
      const shape = this.textureSliceShape;
      const length = shape[0] * shape[1];
      const slices = Math.ceil(data.length / length);

      if (this.textureSlices.length !== slices) {
        throw new Error('Invalid data length in replaceTensor.');
      }
      this.textureSlices.forEach((texture, i) => {
        let oneSlice;
        if (i === slices - 1) {
          let left = data.length % length;
          oneSlice = new this.arrayType(length);
          oneSlice.set(new this.arrayType(data.buffer, data.byteOffset + i * length * data.BYTES_PER_ELEMENT, 
                                                      left === 0 ? length : left), 0);            
        } else {
          oneSlice = new this.arrayType(data.buffer, data.byteOffset + i * length * data.BYTES_PER_ELEMENT, length);
        }
        gl.bindTexture(textureTarget, texture);
        gl.texSubImage2D(textureTarget, 0, 0, 0, shape[1], shape[0], textureFormat, textureType, oneSlice, 0);
      });
    }
  }

  /**
   * Transfer data from webgl texture on GPU to ndarray on CPU
   */
  transferFromGLTexture() {
    if (this.textureSlices) {
      this.tensor = ndarray(new this.arrayType(this.textureShape[0] * this.textureShape[1]), this.textureShape);
      let offset = 0;
      for (let k = 0; k < this.textureSlices.length; k++) {
        // Transfer from textureSlices
        webgl2.bindOutputTexture(this.textureSlices[k], this.textureSliceShape);
        const fragmentData = webgl2.readData(this.textureSliceShape);
        // last fragment may need to be truncated
        if (k === this.textureSlices.length - 1) {
          const truncate = this.tensor.data.length - offset;
          this.tensor.data.set(fragmentData.subarray(0, truncate), offset);
        } else {
          this.tensor.data.set(fragmentData, offset);
        }
        offset += fragmentData.length;
      }
    } else {
      // Transfer from texture
      webgl2.bindOutputTexture(this.texture, this.textureShape);
      this.tensor = ndarray(new this.arrayType([]), this.textureShape);
      // let startTime = performance.now();
      this.tensor.data = webgl2.readData(this.textureShape);
      // console.log(`readData time: ${(performance.now() - startTime).toFixed(2)} ms`)
    }

    if (this.is1D && this.textureShape[0] === 1) {
      // collapse to 1D
      // console.log(`collapse to 1D`)
      this.tensor = squeeze(this.tensor, [0]);
    }
  }

  /**
   * Reshapes data into 2D representation preserving last axis (typically channel axis)
   */
  reshapeTo2D() {
    const axis = this.tensor.shape.length - 1;
    const axisSize = this.tensor.shape[axis];
    const otherAxes = this.tensor.shape.slice(0, axis);
    const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1);

    const reshaped = ndarray(new this.arrayType(otherAxesSize * axisSize), [otherAxesSize, axisSize]);

    const otherAxesData = ndarray(new this.arrayType(otherAxesSize), otherAxes);
    const otherAxesDataRaveled = ndarray(new this.arrayType(otherAxesSize), [otherAxesSize]);
    const axisSlices = Array(this.tensor.shape.length).fill(null);
    for (let n = 0; n < axisSize; n++) {
      axisSlices[axis] = n;
      ops.assign(otherAxesData, this.tensor.pick(...axisSlices));
      otherAxesDataRaveled.data = otherAxesData.data;
      ops.assign(reshaped.pick(null, n), otherAxesDataRaveled);
    }

    this.originalShape = this.tensor.shape;
    this.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.tensor.shape, false, axis);
    this.tensor = reshaped;
    this.is2DReshaped = true;
  }

  /**
   * Reshapes tensor in 2D representation back to original
   *
   * Typically called at the end when data is read back from GPU
   *
   * @param {number} axis
   */
  reshapeFrom2D(axis = -1) {
    if (!this.is2DReshaped) {
      throw new Error('[Tensor] not in reshaped 2D representation.');
    }
    if (!this.originalShape) {
      throw new Error('[Tensor] does not contain originalShape.');
    }

    if (axis < 0) {
      axis = this.originalShape.length + axis;
    }

    // second axis is the channel, or common, axis
    const channelDataSize = this.tensor.shape[0];
    const channels = this.tensor.shape[1];

    const reshaped = ndarray(new this.arrayType(this.originalShape.reduce((a, b) => a * b, 1)), this.originalShape);
    const channelDataRaveled = ndarray(new this.arrayType(channelDataSize), [channelDataSize]);
    const unraveledChannelShape = [...this.originalShape.slice(0, axis), ...this.originalShape.slice(axis + 1)];
    const unraveledChannel = ndarray(
      new this.arrayType(unraveledChannelShape.reduce((a, b) => a * b, 1)),
      unraveledChannelShape
    );
    const axisSlices = Array(this.originalShape.length).fill(null);
    for (let n = 0; n < channels; n++) {
      ops.assign(channelDataRaveled, this.tensor.pick(null, n));
      unraveledChannel.data = channelDataRaveled.data;
      axisSlices[axis] = n;
      ops.assign(reshaped.pick(...axisSlices), unraveledChannel);
    }

    this.tensor = reshaped;
  }
}