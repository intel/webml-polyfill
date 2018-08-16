import Layer from '../../Layer'
import Tensor from '../../Tensor'
import webgl2 from '../../WebGL2'

/**
 * InputLayer layer class
 */
export default class Input extends Layer {
  /**
   * Creates an InputLayer layer
   *
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.name = 'Input'
    this.inputTensor = null;
  }

  /**
   * call
   *
   * @param {number[]} data
   */
  call(data, shape, type) {
    this.inputTensor = null;
    this.inputTensor = new Tensor(data, shape, type);
    if (this.inputTensor.tensor.shape.length <= 2) {
      this.inputTensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
    } else if (this.inputTensor.tensor.shape.length > 2) {
      this.inputTensor.reshapeTo2D();
      this.inputTensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
    } 
    return this.inputTensor;
  }
}
