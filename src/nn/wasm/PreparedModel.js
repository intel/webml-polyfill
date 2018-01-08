import getNNOpsInstance from './NNOps'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from '../Enums'
import * as utils from '../utils'

export default class PreparedModel {
  constructor() {
    this._operations = [];
    this._operands = [];
    this._prepared = false;
  }

  /**
   * Prepare for model execution.
   * 
   * @param {Object} model - 
   */
  async prepare(model) {
    this._operations = model._operations;
    this._operands = model._operands;
    for (let i = 0; i < this._operands.length; ++i) {
      let operand = this._operands[i];
      if (utils.isTensor(operand.type)) {
        operand.value = await this._allocateTensor(operand);
      }
    }
    this._prepared = true;
  }

  /**
   * Launches an asynchronous execution on a prepared model.
   * 
   * @param {*} inputs 
   * @param {*} outputs 
   */
  async execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }
    throw new Error('Not implemented');
  }

  async _allocateTensor(operand) {
    let nn_ops = await getNNOpsInstance();
    let byteLength = utils.sizeOfTensorData(operand.type, operand.dimensions);
    let ptr = nn_ops._malloc(byteLength);
    if (operand.lifetime === OperandLifetime.constant_reference) {
      // Copy data
      let typeEnum = OperandCode.enumValueOf(operand.type);
      if (typeEnum === OperandCode.tensor_float32) {
        nn_ops.HEAPF32.set(operand.value, ptr>>2);
      } else if (typeEnum === OperandCode.tensor_int32) {
        nn_ops.HEAP32.set(operand.value, ptr>>2);
      } else if (typeEnum === OperandCode.tensor_quant8_asymm) {
        nn_ops.HEAPU8.set(operand.value, ptr);
      } else {
        throw new Error(`Operand type ${operand.type} is not supproted`);
      }
    }
    return ptr;
  }
}