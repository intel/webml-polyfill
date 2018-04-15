import _Pool2D from './_Pool2D'

/**
 * AveragePool2D layer class, extends abstract _Pool2D class
 */
export default class AveragePool2D extends _Pool2D {
  /**
   * Creates a AveragePool2D layer
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'AveragePool2D';
    this.poolingFunc = 'average';
  }
}
