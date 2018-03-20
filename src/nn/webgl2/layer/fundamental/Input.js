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
    // this.inputTensor = null;
    if (!this.inputTensor) {
      this.inputTensor = new Tensor(data, shape, type);
      if (this.inputTensor.tensor.shape.length <= 2) {
        this.inputTensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      } else if (this.inputTensor.tensor.shape.length > 2) {
        // console.log(this.inputTensor.tensor.shape.length)
        this.inputTensor.reshapeTo2D();
        this.inputTensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      } 
    } else {
      if (type !== this.inputTensor.arrayType) {
        throw new Error('Invalid data type in InputLayer.');
      }
      if (shape.reduce((i, j) => i * j) !== this.inputTensor.tensor.shape.reduce((i, j) => i * j)) {
        throw new Error('Invalid data shape in InputLayer.');
      }
      this.inputTensor.replaceTensorData(data);
    }
    return this.inputTensor;
  }
}
