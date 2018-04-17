import webgl2 from './WebGL2'

export default class Layer {
  /**
   * Creates a layer
   *
   * @param {Object} [attrs] - layer attributes
   */
  constructor(attrs = {}) {
    this.name = 'Layer';
    this.weights = {};
    this.inputs = attrs.inputs;
    this.outputs = attrs.outputs;
  }

  /**
   * Throws Error, adding layer context info to message
   *
   * @param {string} message
   */
  throwError(message) {
    throw new Error(`[Layer: ${this.name || ''}] ${message}`);
  }

  /**
   * Set layer weights
   *
   * @param {Tensor[]} weightsArr - array of weights which are instances of Tensor
   * @param {boolean} createGLTexture
   */
  setWeights(params, weightsArr, createGLTexture = true) {
    params.forEach((p, i) => {
      this.weights[p] = weightsArr[i];
      if (createGLTexture) {
        this.weights[p].createGLTexture({ type: '2d', format: 'float' });
      }
    })
  }

  /**
   * Layer computational logic
   *
   * @param {Tensor} x
   * @returns {Tensor}
   */
  call(x) {
    this.output = x;
    return this.output;
  }
}
