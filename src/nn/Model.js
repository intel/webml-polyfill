import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'
import * as utils from './utils'

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
    this._operations = [];
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
      numberOfConsumers: 0,
      value: null
    }
    this._operands.push(operand);
    return this._operands.length - 1;
  }

  /**
   * Sets an operand to a constant value.
   * 
   * @param {number} index - The index of the model operand we're setting.
   * @param {TypedArray|number} value - The number for scalar value and typed array for tensor data.
   */
  setOperandValue(index, value) {
    if (index > this._operands.length) {
      throw new Error(`Invalid index ${index}`);
    }
    let operand = this._operands[index];
    if (!this._validateOperandValue(value, operand)) {
      throw new Error(`Invalid value ${value}`);
    }
    operand.value = value;
  }

  /**
   * Add an operation to a model.
   * 
   * @param {number} type - The type of the operation.
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  addOperation(type, inputs, outputs) {
    if (this._completed) {
      throw new Error('addOperation cant modify after model finished');
    }

    if (!this._validateOperationCode(type)) {
      throw new Error(`Invalid operation code ${type}`);
    }
    if (!this._validateOperandList(inputs)) {
      throw new Error(`Invalid inputs ${inputs}`);
    }
    if (!this._validateOperandList(outputs)) {
      throw new Error(`Invalid outputs ${outputs}`);
    }
    let op = {
      type: type,
      inputs: inputs,
      outputs: outputs
    };
    this._operations.push(op);
  }

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
    if (OperandCode.enumValueOf(type) === OperandCode.tensor_quant8_asymm) {
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

  _validateOperandValue(value, operand) {
    let type = operand.type;
    let enumValue = OperandCode.enumValueOf(type);
    if (utils.isTensor(type)) {
      let arrayType = utils.operandCodeToTypedArrayMap.get(enumValue);
      if (value instanceof arrayType) {
        let valueLength = value.length * value.BYTES_PER_ELEMENT;
        let neededLength = utils.sizeOfTensorData(operand.type, operand.dimensions);
        if (valueLength != neededLength) {
          console.error(`Sets ${valueLength} bytes when needing ${neededLength}`);
          return false;
        } else {
          return true;
        }
      } else {
        console.error(`Invalid value type ${typeof value}`);
        return false;
      }
    } else {
      if (typeof value === 'number') {
        return true;
      } else {
        console.error(`Invalid value type ${typeof value}`);
        return false;
      }
    }
  }

  _validateOperationCode(type) {
    let enumValue = OperationCode.enumValueOf(type);
    if (typeof enumValue === 'undefined') {
      return false;
    }
    return true;
  }

  _validateOperandList(list) {
    let ret = true;
    list.forEach(index => {if (index >= this._operands) ret = false;})
    return ret;
  }
}