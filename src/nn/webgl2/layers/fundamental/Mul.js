import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as tensorUtils from '../../utils/tensorUtils'
import webgl2 from '../../WebGL2'
import mulShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/mul'
import { fuseShaderSource } from '../../webgl/fragmentShader/activation/fuse'

 /**
 * Mul layer class
 */
export default class Mul extends Layer {
  /**
   * Creates an mul layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'Mul';
    const { activation = 'NONE' } = attrs;
    this.activation = activation;
    this.fuseActivation = fuseShaderSource[this.activation];
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  call(inputs) {
    if (!this.output) {
      if (inputs[0].is2DReshaped) {
        this.outputShape = inputs[0].originalShape;
        const outputTextureShape = inputs[0].textureShape;
        this.output = new Tensor([], outputTextureShape);
        this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
        this.output.is2DReshaped = true;
        this.output.originalShape = this.outputShape;
        this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.output.originalShape);
      } else {
        const outputTextureShape = inputs[0].textureShape;
        this.output = new Tensor([], outputTextureShape);
        this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
        if (inputs[0].is1D) {
          this.output.is1D = true;
        }
      }
    }

    if (!this.mulProgram) {
      const mulShaderSource = mulShaderSourceFunc(this.fuseActivation);
      this.mulProgram = webgl2.createProgram(mulShaderSource);
    }
    webgl2.runProgram({
      program: this.mulProgram,
      output: this.output,
      inputs: [
        { input: inputs[0], name: 'A' },
        { input: inputs[1], name: 'B' }
      ],
      supportSliceTexture: true
    });
    
    return this.output;
  }
}
