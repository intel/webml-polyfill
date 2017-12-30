export default class Model {
  /**
   * Create an empty model.
   */
  constructor() {}

  /**
   * Indicate that we have finished modifying a model.
   */
  finish() {}

  /**
   * Add an operand to a model.
   * 
   * @param {number} options.type -  The data type, e.g OperandCode.FLOAT32.
   * @param {number[]} options.dimensions - The dimensions of the tensor. It should be nullptr for scalars.
   * @param {number} options.scale - Only for quantized tensors whose value is defined by (value - zeroPoint) * scale.
   * @param {number} options.zeroPoint - Only for quantized tensors whose value is defined by (value - zeroPoint) * scale.
   * @returns {number} - The operand index.
   */
  addOperand(options = {}) {}

  /**
   * Sets an operand to a constant value.
   * 
   * @param {number} index - The index of the model operand we're setting.
   * @param {TypedArray} buffer - The typed array containing the data to use.
   */
  setOperandValue(index, buffer) {}

  /**
   * Add an operation to a model.
   * 
   * @param {number} type - The type of the operation.
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  addOperation(type, inputs, outputs) {}

  /**
   * Specfifies which operands will be the model's inputs and outputs.
   * 
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  identifyInputsAndOutputs(inputs, outputs) {}
}