import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'

export default class Model {
  /**
   * Create an empty model.
   *
   * @param {string} name - The model name.
   */
  constructor(name) {
    this.name = name;
    this._completed = false;
    this._operands = [];
  }

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
  addOperand(options = {}) {
    if (this._completed) {
      throw new Error('addOperand cant modify after model finished');
    }

    if (!this._validateOperandOptions(options)) {
      throw new Error('Invalid options');
    }

    let operand = {
      type: options.type,
      dimensions: options.dimensions,
      scale: options.scale,
      zeroPoint: options.zeroPoint,
      numberOfConsumers: 0
    }
    this._operands.push(operand);
  }

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

  // private methods
  _validateOperandOptions(options) {
    let type = options.type;
    if (!OperandCode.enumValueOf(type)) {
      console.error(`Invalid type ${options.type}`);
      return false;
    }
    if (OperandCode.enumValueOf(type) === OperandCode.enumValueOf('tensor-quant8-asymm')) {
      if (typeof options.zeroPoint === 'undefined') {
        console.error('zeroPoint is undefined');
        return false;
      } else if (options.zeroPoint < 0 || options.zeroPoint > 255) {
        console.error(`Invalid zeroPoint value ${options.zeroPoint}`);
        return false;
      }
      if (options.scale < 0.0) {
        console.error(`Invalid scale ${options.scale}`);
        return false;
      }
    }
    return true;
  }
}