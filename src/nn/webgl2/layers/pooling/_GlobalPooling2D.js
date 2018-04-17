import Layer from '../../Layer'
import Tensor from '../../Tensor'
import webgl2 from '../../WebGL2'
import { globalPoolingShaderSource } from '../../webgl/fragmentShader/pooling/globalPooling'

/**
 * _GlobalPooling2D layer class
 */
export default class _GlobalPooling2D extends Layer {
  /**
   * Creates a _GlobalPooling2D layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = '_GlobalPooling2D';

    const { data_format = 'HWC' } = attrs;
    this.dataFormat = data_format;

    // default pooling function
    // can be `max` or `average`
    this.poolingFunc = 'max';
    this.poolingProgram = webgl2.createProgram(globalPoolingShaderSource);
  }

  /**
   * call
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    if (x.is2DReshaped) {
      this.inputShape = x.originalShape;
    } else {
      // convert to HWC ordering
      if (this.dataFormat === 'CHW') {
        x.tensor = x.tensor.transpose(1, 2, 0);
      }
      this.inputShape = x.tensor.shape;
      x.reshapeTo2D();
      x.createGLTexture({ type: '2d', format: 'float' });
    }
    // create output textures if doesn't already exist
    if (!this.output) {
      this.output = new Tensor([], [1, 1, this.inputShape[2]]);
      this.output.reshapeTo2D();
      this.output.createGLTexture({ type: '2d', format: 'float' });
    }

    // `true` if max pooling, `false` if average pooling
    const isMaxPooling = this.poolingFunc === 'max';
    webgl2.runProgram({
      program: this.poolingProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }],
      uniforms: [
        { value: this.inputShape[0] * this.inputShape[1], type: 'int', name: 'channelDataSize' },
        { value: isMaxPooling, type: 'bool', name: 'isMaxPooling' }
      ]
    });
    return this.output;
  }
}
