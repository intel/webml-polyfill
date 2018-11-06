import Layer from '../../Layer'
import Tensor from '../../Tensor'
import webgl2 from '../../WebGL2'
import * as tensorUtils from '../../utils/tensorUtils'
import ops from 'ndarray-ops'
import matMulShaderSourceFunc from '../../webgl/fragmentShader/arithmetic/matMul'
import { fuseShaderSource } from '../../webgl/fragmentShader/activation/fuse'

/**
 * FullyConnected layer class
 */
export default class FullyConnected extends Layer {
  /**
   * Creates a FullyConnected layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'FullyConnected';
    const {
      activation = 'NONE',
      use_bias = true,
      weights = []
    } = attrs;
    this.activation = activation;
    this.useBias = use_bias;
    this.fuseActivation = fuseShaderSource[this.activation];
    this._setWeights(weights);
  }

  /**
   * Method for setting layer weights. Extends `super` method.
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   */
  _setWeights(weightsArr) {
    const params = this.useBias ? ['kernel', 'bias'] : ['kernel'];
    super.setWeights(params, weightsArr, false);
    this._transposeKernel(); 
    this.weights['kernel'] = this.transposedKernel;
    this.weights['kernel'].createGLTexture({ type: '2d', format: 'float' });
    if (this.useBias) {
      this.weights['bias'].createGLTexture({ type: '2d', format: 'float' });
    }
  }

  /**
   * Transpose weight matrix
   *
   */
  _transposeKernel() {
    const kernelH = this.weights['kernel'].tensor.shape[0];
    const kernelW = this.weights['kernel'].tensor.shape[1];
    this.transposedKernel = new Tensor([], [kernelW, kernelH]);
    for (let i = 0; i < kernelH; i++) {
      ops.assign(
        this.transposedKernel.tensor.pick(null, i), 
        this.weights['kernel'].tensor.pick(i, null)
      );
    }
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    if (!this.output) {
      this.output = new Tensor([], [1, 1, this.weights['kernel'].tensor.shape[1]]);
      this.output.reshapeTo2D();
      this.output.createGLTexture({ type: '2d', format: 'float' });
    }
    const matMulInputs = [{ input: x, name: 'A' }, { input: this.weights['kernel'], name: 'B' }];
    if (this.useBias) {
      matMulInputs.push({ input: this.weights['bias'], name: 'C' });
    }
    if (!this.matMulProgram) {
      this.matMulProgram = webgl2.createProgram(matMulShaderSourceFunc(this.fuseActivation));
    }
    webgl2.runProgram({
      program: this.matMulProgram,
      output: this.output,
      inputs: matMulInputs,
      uniforms: [{ value: this.useBias ? 1 : 0, type: 'bool', name: 'addC' }],
      supportSliceTexture: true
    });
    
    return this.output;
  }
}
