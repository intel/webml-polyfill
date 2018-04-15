import Layer from '../../Layer'

/**
 * Reshape layer class
 * Note there is no concept of batch size in these layers (single-batch).
 */
export default class Reshape extends Layer {
  /**
   * Creates a Reshape layer
   *
   * @param {Object} [attrs] - layer config attributes
   * @param {number[]} [attrs.target_shape]
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'Reshape';

    const { target_shape = [] } = attrs;
    this.targetShape = target_shape;
  }

  /**
   * call
   *
   * @param {Tensor} x
   */
  call(x) {
    return x;
  }
}