import _Pool2D from './_Pool2D'

/**
 * MaxPool2D layer class, extends abstract _Pool2D class
 */
export default class MaxPool2D extends _Pool2D {
  /**
   * Creates a MaxPool2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'MaxPool2D';
    this.poolingFunc = 'max';
  }
}
