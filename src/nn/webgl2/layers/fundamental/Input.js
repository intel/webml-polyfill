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
   * @param {Object} [attrs] - inputsNum: number of inputs
   */
  constructor(attrs = {}) {
    super(attrs)
    this.name = 'Input';
    this.inputsNum = attrs.inputsNum;
    this.inputTensors = Array(this.inputsNum);
  }

  /**
   * call
   *
   * @param {Map} inputs - input map with value: inputBuffers and indexes identifying the input operands.
   */
  call(inputs, shape, type) {
    inputs.forEach((input, i) => {
      if (!this.inputTensors[i]) {
        this.inputTensors[i] = new Tensor(input.buffer, shape, type);
        if (this.inputTensors[i].tensor.shape.length <= 2) {
          this.inputTensors[i].createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
        } else if (this.inputTensors[i].tensor.shape.length > 2) {
          this.inputTensors[i].reshapeTo2D();
          this.inputTensors[i].createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
        } 
      } else {	
        if (type !== this.inputTensors[i].arrayType) {	
          this.throwError('Invalid data type in InputLayer.');	
        }	
        if (shape.reduce((a, b) => a * b) !== this.inputTensors[i].tensor.shape.reduce((a, b) => a * b)) {	
          this.throwError('Invalid data shape in InputLayer.');	
        }	
        this.inputTensors[i].replaceTensorData(input.buffer);	
      }
    });
    return this.inputTensors;
  }
}
