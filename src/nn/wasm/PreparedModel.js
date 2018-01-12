import getNNOpsInstance from './NNOps'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from '../Enums'
import * as utils from '../utils'
import { product } from '../utils';

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
    this._nn_ops = await getNNOpsInstance();
    this._operations = model._operations;
    for (let i = 0; i < model._operands.length; ++i) {
      let operand = model._operands[i];
      let runtimeOperand = {};
      runtimeOperand.type = operand.type;
      if (utils.isTensor(operand.type)) {
        runtimeOperand.value = this._allocateTensor(operand);
        runtimeOperand.shape = this._allocateShape(operand);
      } else {
        runtimeOperand.value = operand.value;
      }
      this._operands.push(runtimeOperand);
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
    let op = operation.type;
    let inputs = operation.inputs;
    let outputs = operation.outputs;
    let operands = this._operands;
    let success;

    function allParametersPresent(requiredIns, requiredOuts) {
      function verify(requiredCount, indexes, type) {
        let actualCount = indexes.length;
        if (requiredCount !== actualCount) {
          throw new Error(`Operation ${op} requires ${requiredCount} ${type} operands, but got ${actualCount}.`);
        }
        indexes.forEach(index => {
          if (operands[index].value === null || operands[index].lifetime === OperandLifetime.NO_VALUE) {
            throw new Error(`Operation ${op} ${type} operand ${index} is required but missing.`);
          }
        })
      }
      verify(requiredIns, inputs, 'in');
      verify(requiredOuts, outputs, 'out');
    }

    const FuseCodeMap = new Map([
      [FuseCode.NONE, nn_ops.NONE],
      [FuseCode.RELU, nn_ops.RELU],
      [FuseCode.RELU1, nn_ops.RELU1],
      [FuseCode.RELU6, nn_ops.RELU6],
    ]);

    function calculateExplicitPadding(inSize, stride, filterSize, paddingCode) {
      let paddingHead = 0;
      let paddingTail = 0;

      if (paddingCode == PaddingCode.SAME) {
        let outSize = (inSize + stride - 1) / stride;
        let tmp = (outSize - 1) * stride + filterSize;
        if (tmp > inSize) {
          paddingHead = (tmp - inSize) / 2;
          paddingTail = (tmp - inSize) - paddingHead;
        }
      }

      return [paddingHead, paddingTail];
    }
      

    switch(op) {
      case OperationCode.ADD: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseCodeMap.get(operands[inputs[2]].value);
        let out = operands[outputs[0]];
        success = nn_ops.addMulPrepare(in1.shape, in2.shape, out.shape);
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
      case OperationCode.MUL: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseCodeMap.get(operands[inputs[2]].value);
        let out = operands[outputs[0]];
        success = nn_ops.addMulPrepare(in1.shape, in2.shape, out.shape);
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
      case OperationCode.CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of CONV_2D');
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let paddingLeft, paddingRight;
        let paddingTop, paddingBottom;
        let strideWidth, strideHeight;
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value;
          paddingRight = operands[inputs[i++]].value;
          paddingTop = operands[inputs[i++]].value;
          paddingBottom = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);
        } else {
          let paddingCode = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);

          let inputWidth = input.shape.dimensions[2];
          let inputHeight = input.shape.dimensions[1];
          let filterWidth = filter.shape.dimensions[2];
          let filterHeight = filter.shape.dimensions[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];
        success = nn_ops.convPrepare(input.shape, filter.shape, bias.shape,
                                     paddingLeft, paddingRight,
                                     paddingTop, paddingBottom,
                                     strideWidth, strideHeight,
                                     output.shape);
        if (!success) {
          throw new Error('convPrepare fails');
        }
        success = nn_ops.convFloat32(input.value, input.shape,
                                      filter.value, filter.shape,
                                      bias.value, bias.shape,
                                      paddingLeft, paddingRight,
                                      paddingTop, paddingBottom,
                                      strideWidth, strideHeight, activation,
                                      output.value, output.shape);
        if (!success) {
          throw new Error('convFloat32 fails');
        }
      } break;
      case OperationCode.DEPTHWISE_CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error('Invalid parameters number of DEPTHWISE_CONV_2D');
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let paddingLeft, paddingRight;
        let paddingTop, paddingBottom;
        let strideWidth, strideHeight;
        let depthMultipler;
        let activation;
        if (inCount === 11) {
          paddingLeft = operands[inputs[i++]].value;
          paddingRight = operands[inputs[i++]].value;
          paddingTop = operands[inputs[i++]].value;
          paddingBottom = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          depthMultipler = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);
        } else {
          let paddingCode = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          depthMultipler = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);

          let inputWidth = input.shape.dimensions[2];
          let inputHeight = input.shape.dimensions[1];
          let filterWidth = filter.shape.dimensions[2];
          let filterHeight = filter.shape.dimensions[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];
        success = nn_ops.depthwiseConvPrepare(input.shape, filter.shape, bias.shape,
                                              paddingLeft, paddingRight,
                                              paddingTop, paddingBottom,
                                              strideWidth, strideHeight,
                                              output.shape);
        if (!success) {
          throw new Error('depthwiseConvPrepare fails');
        }
        success = nn_ops.depthwiseConvFloat32(input.value, input.shape,
                                              filter.value, filter.shape,
                                              bias.value, bias.shape,
                                              paddingLeft, paddingRight,
                                              paddingTop, paddingBottom,
                                              strideWidth, strideHeight,
                                              depthMultipler, activation,
                                              output.value, output.shape);
        if (!success) {
          throw new Error('depthwiseConvFloat32 fails');
        }
      } break;
      case OperationCode.AVERAGE_POOL_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of AVERAGE_POOL_2D');
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let paddingLeft, paddingRight;
        let paddingTop, paddingBottom;
        let strideWidth, strideHeight;
        let filterWidth, filterHeight;
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value;
          paddingRight = operands[inputs[i++]].value;
          paddingTop = operands[inputs[i++]].value;
          paddingBottom = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          filterWidth = operands[inputs[i++]].value;
          filterHeight = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);
        } else {
          let paddingCode = operands[inputs[i++]].value;
          strideWidth = operands[inputs[i++]].value;
          strideHeight = operands[inputs[i++]].value;
          filterWidth = operands[inputs[i++]].value;
          filterHeight = operands[inputs[i++]].value;
          activation = FuseCodeMap.get(operands[inputs[i++]].value);

          let inputWidth = input.shape.dimensions[2];
          let inputHeight = input.shape.dimensions[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];
        success = nn_ops.genericPoolingPrepare(input.shape,
                                               paddingLeft, paddingRight,
                                               paddingTop, paddingBottom,
                                               strideWidth, strideHeight,
                                               filterWidth, filterHeight,
                                               output.shape);
        if (!success) {
          throw new Error('genericPoolingPrepare fails');
        }
        success = nn_ops.averagePoolFloat32(input.value, input.shape,
                                            paddingLeft, paddingRight,
                                            paddingTop, paddingBottom,
                                            strideWidth, strideHeight,
                                            filterWidth, filterHeight, activation,
                                            output.value, output.shape);
        if (!success) {
          throw new Error('averagePoolFloat32 fails');
        }
      } break;
      case OperationCode.SOFTMAX: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let beta = operands[inputs[1]].value;
        if (beta <= 0.0) {
          throw new Error('beta must be positive for SOFTMAX');
        }
        let output = operands[outputs[0]];
        success = nn_ops.genericActivationPrepare(input.shape, output.shape);
        if (!success) {
          throw new Error('genericActivationPrepare fails');
        }
        success = nn_ops.softmaxFloat32(input.value, input.shape, beta, output.value, output.shape);
        if (!success) {
          throw new Error('softmaxFloat32 fails');
        }
      } break;
      case OperationCode.RESHAPE: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let targetShape = operands[inputs[1]];
        let targetShapeBufferLength = product(targetShape.shape.dimensions);

        let output = operands[outputs[0]];
        success = nn_ops.reshapePrepare(input.shape, targetShape.value, targetShapeBufferLength, output.shape);
        if (!success) {
          throw new Error('reshapePrepare fails');
        }
        success = nn_ops.reshapeGeneric(input.value, input.shape, output.value, output.shape);
        if (!success) {
          throw new Error('reshapeGeneric fails');
        }
      } break;
      default: {
        throw new Error(`Operation ${op} is not supported`);
      }
    }
  }

  _setTensorData(type, ptr, data) {
    const nn_ops = this._nn_ops;
    if (type === OperandCode.TENSOR_FLOAT32) {
      nn_ops.HEAPF32.set(data, ptr>>2);
    } else if (type === OperandCode.TENSOR_INT32) {
      nn_ops.HEAP32.set(data, ptr>>2);
    } else if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
      nn_ops.HEAPU8.set(data, ptr);
    } else {
      throw new Error(`Operand type ${type} is not supproted`);
    }
  }

  _getTensorData(type, ptr, buffer) {
    const nn_ops = this._nn_ops;
    let view;
    if (type === OperandCode.TENSOR_FLOAT32) {
      view = new Float32Array(nn_ops.HEAPF32.buffer, ptr, buffer.length);
    } else if (type === OperandCode.TENSOR_INT32) {
      view = new Int32Array(nn_ops.HEAP32.buffer, ptr, buffer.length);
    } else if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
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
    if (operand.lifetime === OperandLifetime.CONSTANT_REFERENCE) {
      this._setTensorData(operand.type, ptr, operand.value);
    }
    return ptr;
  }

  _allocateShape(operand) {
    const nn_ops = this._nn_ops;
    const OperandTypeMap = new Map([
      [OperandCode.TENSOR_FLOAT32, nn_ops.TENSOR_FLOAT32],
      [OperandCode.TENSOR_INT32, nn_ops.TENSOR_INT32],
      [OperandCode.TENSOR_QUANT8_ASYMM, nn_ops.TENSOR_QUANT8_ASYMM]
    ]);
    let shape = new nn_ops.Shape;
    shape.type = OperandTypeMap.get(operand.type);
    shape.dimensions = operand.dimensions;
    return shape;
  }
}