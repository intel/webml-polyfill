import getNNOpsInstance from './NNOps'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from '../Enums'
import * as utils from '../utils'

export default class PreparedModel {
  constructor() {
    this._operations = [];
    this._operands = [];
    this._prepared = false;
    this._nn_ops = null;
  }

  /**
   * Prepare for model execution.
   * 
   * @param {Object} model - A model object built by user.
   */
  async prepare(model) {
    this._operations = model._operations;
    this._operands = model._operands;
    this._nn_ops = await getNNOpsInstance();
    for (let i = 0; i < this._operands.length; ++i) {
      let operand = this._operands[i];
      if (utils.isTensor(operand.type)) {
        operand.value = this._allocateTensor(operand);
        operand.shape = this._allocateShape(operand);
      }
    }
    this._prepared = true;
  }

  /**
   * Launches an asynchronous execution on a prepared model.
   * 
   * @param {Array} inputs - Inputs provided by user.
   * @param {Array} outputs - Outputs will receive results.
   */
  async execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }

    inputs.forEach(input => {
      let operand = this._operands[input.index];
      let buffer = input.buffer;
      this._setTensorData(operand.type, operand.value, buffer);
    });

    this._operations.forEach(operation => {
      this._executeOperation(operation);
    });

    outputs.forEach(output => {
      let operand = this._operands[output.index];
      let buffer = output.buffer;
      this._getTensorData(operand.type, operand.value, buffer);
    });
  }

  _executeOperation(operation) {
    const nn_ops = this._nn_ops;
    let op = OperationCode.enumValueOf(operation.type);
    let inputs = operation.inputs;
    let outputs = operation.outputs;
    let operands = this._operands;

    function allParametersPresent(requiredIns, requiredOuts) {
      function verify(requiredCount, indexes, type) {
        let actualCount = indexes.length;
        if (requiredCount !== actualCount) {
          throw new Error(`Operation ${op} requires ${requiredCount} ${type} operands, but got ${actualCount}.`);
        }
        indexes.forEach(index => {
          if (operands[index].value === null || operands[index].lifetime === OperandLifetime.no_value) {
            throw new Error(`Operation ${op} ${type} operand ${index} is required but missing.`);
          }
        })
      }
      verify(requiredIns, inputs, 'in');
      verify(requiredOuts, outputs, 'out');
    }

    const FuseCodeMap = new Map([
      [FuseCode.none, nn_ops.NONE],
      [FuseCode.relu, nn_ops.RELU],
      [FuseCode.relu1, nn_ops.RELU1],
      [FuseCode.relu6, nn_ops.RELU6],
    ]);

    switch(op) {
      case OperationCode.add: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseCodeMap.get(operands[inputs[2]].value);
        let out = operands[outputs[0]];
        let success = nn_ops.addMulPrepare(in1.shape, in2.shape, out.shape);
        if (!success) {
          throw new Error('addMulPrepare fails');
        }
        success = nn_ops.addFloat32(in1.value, in1.shape,
                                    in2.value, in2.shape,
                                    activation,
                                    out.value, out.shape);
        if (!success) {
          throw new Error('addFloat32 fails');
        }
      } break;
      case OperationCode.mul: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseCodeMap.get(operands[inputs[2]].value);
        let out = operands[outputs[0]];
        let success = nn_ops.addMulPrepare(in1.shape, in2.shape, out.shape);
        if (!success) {
          throw new Error('addMulPrepare fails');
        }
        success = nn_ops.mulFloat32(in1.value, in1.shape,
                                    in2.value, in2.shape,
                                    activation,
                                    out.value, out.shape);
        if (!success) {
          throw new Error('mulFloat32 fails');
        }
      } break;
      default: {
        throw new Error(`Operation ${op} is not supported`);
      }
    }
  }

  _setTensorData(type, ptr, data) {
    const nn_ops = this._nn_ops;
    let typeEnum = OperandCode.enumValueOf(type);
    if (typeEnum === OperandCode.tensor_float32) {
      nn_ops.HEAPF32.set(data, ptr>>2);
    } else if (typeEnum === OperandCode.tensor_int32) {
      nn_ops.HEAP32.set(data, ptr>>2);
    } else if (typeEnum === OperandCode.tensor_quant8_asymm) {
      nn_ops.HEAPU8.set(data, ptr);
    } else {
      throw new Error(`Operand type ${type} is not supproted`);
    }
  }

  _getTensorData(type, ptr, buffer) {
    const nn_ops = this._nn_ops;
    let typeEnum = OperandCode.enumValueOf(type);
    let view;
    if (typeEnum === OperandCode.tensor_float32) {
      view = new Float32Array(nn_ops.HEAPF32.buffer, ptr, buffer.length);
    } else if (typeEnum === OperandCode.tensor_int32) {
      view = new Int32Array(nn_ops.HEAP32.buffer, ptr, buffer.length);
    } else if (typeEnum === OperandCode.tensor_quant8_asymm) {
      view = new Uint8Array(nn_ops.HEAPU8.buffer, ptr, buffer.length);
    } else {
      throw new Error(`Operand type ${type} is not supproted`);
    }
    buffer.set(view);
  }

  _allocateTensor(operand) {
    const nn_ops = this._nn_ops;
    let byteLength = utils.sizeOfTensorData(operand.type, operand.dimensions);
    let ptr = nn_ops._malloc(byteLength);
    if (operand.lifetime === OperandLifetime.constant_reference) {
      this._setTensorData(operand.type, ptr, operand.value);
    }
    return ptr;
  }

  _allocateShape(operand) {
    const nn_ops = this._nn_ops;
    const OperandTypeMap = new Map([
      [OperandCode.tensor_float32, nn_ops.TENSOR_FLOAT32],
      [OperandCode.tensor_int32, nn_ops.tensor_int32],
      [OperandCode.tensor_quant8_asymm, nn_ops.TENSOR_QUANT8_ASYMM]
    ]);
    let shape = new nn_ops.Shape;
    shape.type = OperandTypeMap.get(OperandCode.enumValueOf(operand.type));
    shape.dimensions = operand.dimensions;
    return shape;
  }
}