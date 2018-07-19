import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as activations from '../../webgl/fragmentShader/activation'
import webgl2 from '../../WebGL2'

/**
 * Activation layer class
 */
export default class Activation extends Layer {
  /**
   * Creates a avtivation layer

   */
  constructor(attrs = {}) {
    super(attrs);
    const {
      activation = 'NONE'
    } = attrs;
    this.name = activation;
    if (this.name !== 'NONE' && !webgl2.activationProgram) {
      if (this.name === 'softmax') {
        this.activationProgram = webgl2.createProgram(activations['softmax'](attrs.beta));
      } else {
        this.activationProgram = webgl2.createProgram(activations[this.name]);
      }
    }
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    if (this.name !== 'NONE') {
      if (!this.output) {
        this.output = new Tensor([], x.tensor.shape);
        this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      }
      webgl2.runProgram({
        program: this.activationProgram,
        output: this.output,
        inputs: [{ input: x, name: 'x' }],
        supportSliceTexture: true
      });
      return this.output;
    } else {
      return x;
    }
  }
}
