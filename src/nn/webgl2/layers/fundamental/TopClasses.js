import Layer from '../../Layer'
import Tensor from '../../Tensor'
import webgl2 from '../../WebGL2'
import topClassesShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/topClasses'

/**
 * TopClasses layer class
 */
export default class TopClasses extends Layer {
  /**
   * Creates a TopClasses layer

   */
  constructor(attrs = {}) {
    super(attrs);
    const {
      numTopClasses = 3
    } = attrs
    this.name = 'TopClasses';
    this.numTopClasses = numTopClasses;
    this.output = null;
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    if (!this.output) {
      // console.log(x.tensor.shape)
      // x.tensor.shape = [1, length]
      this.output = new Tensor([], x.tensor.shape)
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true })
    }
    if (!this.topClassesProgram) {
      const topClassesShaderSource = topClassesShaderSourceFunc(this.numTopClasses, x.tensor.shape[1]);
      this.topClassesProgram = webgl2.createProgram(topClassesShaderSource)
    }
    // console.log(this.output);
    // console.log(x);
    webgl2.runProgram({
      program: this.topClassesProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }],
      supportSliceTexture: true
    });
    let out = webgl2.readData([1, this.numTopClasses * 2])
    return out;
  }

}
