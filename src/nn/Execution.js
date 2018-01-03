import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from './Enums'

export default class Execution {
  /**
   * Create an Execution to apply the given compilation.
   * 
   * @param {Compilation} compilation 
   */
  constructor(compilation) {
    this._compilation = compilation;
    this._model = compilation._model;
    this._executor = null;
  }

  /**
   * Associate a user data with an input of the model of the Execution.
   * 
   * @param {number} index - The index of the input argument we are setting.
   * @param {TypedArray} value - The typed array containing the data.
   */
  setInput(index, value) {
    let model = this._model;
    if (index > model._inputs.length) {
      throw new Error(`Invalid index ${index}`);
    }
    let inputIndex = model._inputs[index];
    if (inputIndex > model._operands.length) {
      throw new Error(`Invalid input index ${inputIndex}`);
    }
    let operand = model._operands[inputIndex];
    if (!model._validateOperandValue(value, operand)) {
      throw new Error(`Invalid value ${value}`);
    }
    if (operand.lifetime !== OperandLifetime.model_input) {
      throw new Error(`Invalid operand lifetime ${operand.lifetime}`);
    }
    operand.value = value;
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
    if (operand.lifetime !== OperandLifetime.model_output) {
      throw new Error(`Invalid operand lifetime ${operand.lifetime}`);
    }
    operand.value = buffer;
  }

  /**
   * Schedule evaluation of the execution.
   */
  async startCompute() {
    return new Error('Not implemented');
  }
}