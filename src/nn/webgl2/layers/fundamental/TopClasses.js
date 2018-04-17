import Layer from '../../Layer'
import Tensor from '../../Tensor'
import webgl2 from '../../WebGL2'
import topClassesShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/topClasses'
import reduceClassesShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/reduceClasses'

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
      numTopClasses = 1,
      reduceNums = 10
    } = attrs
    this.name = 'TopClasses';
    this.numTopClasses = numTopClasses;
    this.output = null;
    this.outArray = [];
    this.programArray = [];
    this.reduceNums = reduceNums;
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    if (this.numTopClasses === 1) {
      let index = 0;
      let reduceLen = Math.ceil(x.tensor.shape[1] / this.reduceNums);
      let input = x;
      while (reduceLen > 0) {
        // console.log(`index: ${index}`);
        // console.log(`reduceLen: ${reduceLen}`);
        if (!this.outArray[index]) {
          this.outArray[index] = new Tensor([], [2, reduceLen]);
          this.outArray[index].createGLTexture({ type: '2d', format: 'float'});
        }
        if (!this.programArray[index]) {
          const reduceClassesShaderSource = reduceClassesShaderSourceFunc(this.reduceNums, reduceLen, index, input.tensor.shape[1]);
          this.programArray[index] = webgl2.createProgram(reduceClassesShaderSource);
        }
        webgl2.runProgram({
          program: this.programArray[index],
          output: this.outArray[index],
          inputs: [{ input: input, name: 'x' }],
          supportSliceTexture: true
        });
        // let out = webgl2.readData([2, reduceLen]);
        // console.log(`out: ${out}`)
        if (reduceLen < 2) {
          break;
        }
        input = this.outArray[index];
        ++index;
        reduceLen = Math.ceil(reduceLen / this.reduceNums);
      }
      // let start = performance.now();
      let out = webgl2.readData([2, reduceLen]);
      // console.log('Read data from GPU time', (performance.now() - start).toFixed(2));
      // console.log(`out: ${out}`);
      return out;
    } else {
      if (!this.output) {
        // x.tensor.shape = [1, length]
        this.output = new Tensor([], x.tensor.shape)
        this.output.createGLTexture({ type: '2d', format: 'float'})
      }
      if (!this.topClassesProgram) {
        const topClassesShaderSource = topClassesShaderSourceFunc(this.numTopClasses, x.tensor.shape[1]);
        this.topClassesProgram = webgl2.createProgram(topClassesShaderSource);
      }
      webgl2.runProgram({
        program: this.topClassesProgram,
        output: this.output,
        inputs: [{ input: x, name: 'x' }],
        supportSliceTexture: true
      });
      // let start = performance.now();
      let out = webgl2.readData([1, this.numTopClasses * 2]);
      // console.log('Read data from GPU time', (performance.now() - start).toFixed(2));
      return out;
    }
  }
}
