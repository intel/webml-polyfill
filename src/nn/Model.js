import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime, ResultCode} from './Enums'
import * as utils from './utils'
import Compilation from './Compilation';

export default class Model {
  /**
   * Create an empty model.
   */
  constructor(options = {}) {
    this._completed = false;
    this._operands = [];
    this._operations = [];
    this._inputs = null;
    this._outputs = null;
    this._backend = options.backend;
  }

  /**
   * Create a compilation from model.
   * 
   * @returns {Compilation} - the compilation object.
   */
  async createCompilation() {
    if (!this._completed) {
      throw new Error('Model is not finished');
    }
    return new Compilation(this);
  }

  /**
   * Indicate that we have finished modifying a model.
   */
  async finish() {
    if (this._completed) {
      throw new Error('finish called more than once');
    }

    if (this._inputs === null || this._outputs === null) {
      throw new Error('No inputs or outputs of this model');
    }

    this._sortIntoRunOrder();
    this._completed = true;
    return ResultCode.NO_ERROR;
  }

  /**
   * Add an operand to a model.
   * 
   * @param {number} options.type -  The data type, e.g OperandCode.FLOAT32.
   * @param {number[]} options.dimensions - The dimensions of the tensor. It should be nullptr for scalars.
   * @param {number} options.scale - Only for quantized tensors whose value is defined by (value - zeroPoint) * scale.
   * @param {number} options.zeroPoint - Only for quantized tensors whose value is defined by (value - zeroPoint) * scale.
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
      lifetime: OperandLifetime.TEMPORARY_VARIABLE,
      value: null
    }
    this._operands.push(operand);
    // return ResultCode.NO_ERROR;
  }

  /**
   * Sets an operand to a constant value.
   * 
   * @param {number} index - The index of the model operand we're setting.
   * @param {TypedArray} value - The typed array containing data.
   */
  setOperandValue(index, value) {
    if (index > this._operands.length) {
      throw new Error(`Invalid index ${index}`);
    }
    let operand = this._operands[index];
    if (!this._validateOperandValue(value, operand)) {
      throw new Error(`Invalid value ${value}`);
    }
    if (utils.isTensor(operand.type)) {
      operand.lifetime = OperandLifetime.CONSTANT_REFERENCE;
    } else {
      operand.lifetime = OperandLifetime.CONSTANT_COPY;
    }
    operand.value = value;
    // return ResultCode.NO_ERROR;
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
    inputs.forEach(i => {
      this._operands[i].numberOfConsumers += 1;
    });
    this._operations.push(op);
    // return ResultCode.NO_ERROR;
  }

  /**
   * Specfifies which operands will be the model's inputs and outputs.
   * 
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  identifyInputsAndOutputs(inputs, outputs) {
    if (!this._validateOperandList(inputs)) {
      throw new Error(`Invalid inputs ${inputs}`);
    }
    if (!this._validateOperandList(outputs)) {
      throw new Error(`Invalid outputs ${outputs}`);
    }
    this._inputs = inputs;
    this._inputs.forEach(i => {
      this._operands[i].lifetime = OperandLifetime.MODEL_INPUT;
    })
    this._outputs = outputs;
    this._outputs.forEach(i => {
      this._operands[i].lifetime = OperandLifetime.MODEL_OUTPUT;
    })
    // return ResultCode.NO_ERROR;
  }

  // private methods
  _validateOperandOptions(options) {
    let type = options.type;
    if (!utils.validateEnum(type, OperandCode)) {
      console.error(`Invalid type ${options.type}`);
      return false;
    }
    if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
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
    let arrayType = utils.operandCodeToTypedArrayMap.get(type);
    if (value instanceof arrayType) {
      let valueLength = value.length * value.BYTES_PER_ELEMENT;
      let neededLength;
      if (utils.isTensor(type)) {
        neededLength = utils.sizeOfTensorData(type, operand.dimensions);
      } else {
        neededLength = utils.sizeOfScalarData(type);
      }
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
  }

  _validateOperationCode(type) {
    return utils.validateEnum(type, OperationCode);
  }

  _validateOperandList(list) {
    let ret = true;
    list.forEach(index => {if (index >= this._operands) ret = false;})
    return ret;
  }

  _sortIntoRunOrder() {
    let opsReadyToRun = [];
    let runOrder = [];
    let unknownInputCount = new Array(this._operations.length);
    unknownInputCount.fill(0);
    let operandToOperations = new Map();
    this._operations.forEach((operation, operationIndex) => {
      let inputs = operation.inputs;
      inputs.forEach(operandIndex => {
        let lifetime = this._operands[operandIndex].lifetime;
        if (lifetime === OperandLifetime.TEMPORARY_VARIABLE || lifetime === OperandLifetime.MODEL_OUTPUT) {
          unknownInputCount[operationIndex] += 1;
          if (!operandToOperations.has(operandIndex)) {
            operandToOperations.set(operandIndex, [operationIndex]);
          } else {
            let array = operandToOperations.get(operandIndex);
            array.push(operationIndex)
            operandToOperations.set(operandIndex, array);
          }
        }
      });
      if (unknownInputCount[operationIndex] === 0) {
        opsReadyToRun.push(operationIndex)
      }
    });

    while(opsReadyToRun.length > 0) {
      let opIndex = opsReadyToRun.pop();
      let operation = this._operations[opIndex];
      runOrder.push(operation);

      operation.outputs.forEach(operandIndex => {
        if (operandToOperations.has(operandIndex)) {
          operandToOperations.get(operandIndex).forEach(operationIndex => {
            unknownInputCount[operationIndex] -= 1;
            if (unknownInputCount[operationIndex] === 0) {
              opsReadyToRun.push(operationIndex);
            }
          });
        }
      });
    }

    this._operations = runOrder;
  }
}