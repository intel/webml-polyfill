import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime, ResultCode} from './Enums'

import PreparedModel from './wasm/PreparedModel'

export default class Execution {
  /**
   * Create an Execution to apply the given compilation.
   * 
   * @param {Compilation} compilation 
   */
  constructor(compilation) {
    if (typeof compilation === 'undefined') {
      throw new Error('Invalid argument');
    }
    this._preparedModel = compilation._preparedModel;
    this._model = compilation._model;
    this._inputs = new Map();
    this._outputs = new Map();
  }

  /**
   * Associate a user data with an input of the model of the Execution.
   * 
   * @param {number} index - The index of the input argument we are setting.
   * @param {TypedArray} buffer - The typed array containing the data.
   */
  setInput(index, buffer) {
    let model = this._model;
    if (index > model._inputs.length) {
      throw new Error(`Invalid index ${index}`);
    }
    let inputIndex = model._inputs[index];
    if (inputIndex > model._operands.length) {
      throw new Error(`Invalid input index ${inputIndex}`);
    }
    let operand = model._operands[inputIndex];
    if (!model._validateOperandValue(buffer, operand)) {
      throw new Error(`Invalid value ${buffer}`);
    }
    if (operand.lifetime !== OperandLifetime.MODEL_INPUT) {
      throw new Error(`Invalid operand lifetime ${operand.lifetime}`);
    }
    let tensor = {
      index: inputIndex,
      buffer: buffer
    }
    this._inputs.set(index, tensor);
    return ResultCode.NO_ERROR;
  }

  /**
   * Associate a user buffer with an output of the model of the Execution.
   * 
   * @param {number} index - The index of output.
   * @param {TypedArray} buffer - The typed array to receive the output data.
   */
  setOutput(index, buffer) {
    let model = this._model;
    if (index > model._outputs.length) {
      throw new Error(`Invalid index ${index}`);
    }
    let outputIndex = model._outputs[index];
    if (outputIndex > model._operands.length) {
      throw new Error(`Invalid output index ${outputIndex}`);
    }
    let operand = model._operands[outputIndex];
    if (!model._validateOperandValue(buffer, operand)) {
      throw new Error(`Invalid value ${buffer}`);
    }
    if (operand.lifetime !== OperandLifetime.MODEL_OUTPUT) {
      throw new Error(`Invalid operand lifetime ${operand.lifetime}`);
    }
    let tensor = {
      index: outputIndex,
      buffer: buffer
    }
    this._outputs.set(index, tensor);
    return ResultCode.NO_ERROR;
  }

  /**
   * Schedule evaluation of the execution.
   */
  async startCompute() {
    await this._preparedModel.execute(this._inputs, this._outputs);
    return ResultCode.NO_ERROR;
  }
}