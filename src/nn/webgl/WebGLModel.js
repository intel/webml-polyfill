import * as utils from '../utils'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from '../Enums'
import * as tf from '@tensorflow/tfjs-core';

export default class WebGLModel {
  /**
   * Create WebGLModel class in nn/Model.js
   *
   * @param {Object} model - Model from nn/Model.js
   */
  constructor(model) {
    this._model = model;
    this._operations = model._operations;
    this._operands = [];
    this._prepared = false;

    if (tf.ENV.backend.floatPrecision() === 16) {
      console.warn('The current floating point operation precision is only 16-bit');
    }
  }

  /**
   * Called in nn/Compilation.js
   *
   */
  prepareModel() {
    this._model._operands.forEach(operand => {
      if (utils.isTensor(operand.type)) {
        const type = this._getOperandType(operand.type);
        if (operand.value !== null) {   
          // constant tensor
          this._operands.push(tf.tensor(operand.value, operand.dimensions, type));
        } else {                        
          // variable tensor 
          const zeroTensor = tf.zeros(operand.dimensions, type);
          this._operands.push(tf.variable(zeroTensor));
          zeroTensor.dispose();
        }
      } else {
        this._operands.push(operand);   
      }
    }); 
    this._changeWeightsFormat();
    this._prepared = true;
  }

  /**
   * Called in nn/Execution.js
   *
   * @param {Map} inputs 
   * @param {Map} outputs 
   */
  execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }

    inputs.forEach(input => {
      const operand = this._operands[input.index];
      const inputTensor = tf.tensor(input.buffer, operand.shape, operand.dtype);
      operand.assign(inputTensor);
      inputTensor.dispose();
    });

    this._operations.forEach(operation => {
      tf.tidy(() => {
        this._executeOperation(operation);
      });
    });

    outputs.forEach(output => {
      const operand = this._operands[output.index];  
      output.buffer.set(operand.dataSync());
    });
  }

  _executeOperation(operation) {
    const op = operation.type;
    const inputs = operation.inputs;
    const outputs = operation.outputs;
    const operands = this._operands;

    const FuseFunctionMap = new Map([
      [FuseCode.NONE, x => x],
      [FuseCode.RELU, tf.relu],
      [FuseCode.RELU1, x => tf.clipByValue(x, -1, 1)],
      [FuseCode.RELU6, x => tf.clipByValue(x, 0, 6)]
    ]);

    const PaddingCodeMap = new Map([
      [PaddingCode.SAME, 'same'],
      [PaddingCode.VALID, 'valid']
    ]);

    switch(op) {
      case OperationCode.ADD: {
        const in1 = operands[inputs[0]];
        const in2 = operands[inputs[1]];
        const activation = FuseFunctionMap.get(operands[inputs[2]].value[0]);
        const output = operands[outputs[0]];
        output.assign(activation(tf.add(in1, in2)));
      } break;
      case OperationCode.MUL: {
        const in1 = operands[inputs[0]];
        const in2 = operands[inputs[1]];
        const activation = FuseFunctionMap.get(operands[inputs[2]].value[0]);
        const output = operands[outputs[0]];
        output.assign(activation(tf.mul(in1, in2)));
      } break;
      case OperationCode.CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of CONV_2D');
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        let strideW, strideH;
        let activation;
        if (inCount === 7) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.conv2d(filter, [strideH, strideW], padding).add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                   .conv2d(filter, [strideH, strideW], 'valid').add(bias)));
        }
      } break;
      case OperationCode.DEPTHWISE_CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error('Invalid parameters number of DEPTHWISE_CONV_2D');
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        let strideW, strideH;
        let depthMultipler;
        let activation;
        let paddedInput = input;
        const inputInChannels = input.shape[3];
        const filterInChannels = filter.shape[2];
        if (inputInChannels < filterInChannels) {
          paddedInput = input.pad([[0, 0], [0, 0], [0, 0], [0, filterInChannels - inputInChannels]]);
        }
        if (inCount === 8) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              paddedInput.depthwiseConv2D(filter, [strideH, strideW], padding).add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              paddedInput.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                         .depthwiseConv2D(filter, [strideH, strideW], 'valid').add(bias)));
        }
      } break;
      case OperationCode.ATROUS_CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of ATROUS_CONV_2D');
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        const strides = [1, 1];
        let dilationW, dilationH;
        let activation;
        if (inCount === 7) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          dilationW = operands[inputs[i++]].value[0];
          dilationH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.conv2d(filter, strides, padding, 'NHWC', [dilationH, dilationW]).add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          dilationW = operands[inputs[i++]].value[0];
          dilationH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                   .conv2d(filter, strides, 'valid', 'NHWC', [dilationH, dilationW]).add(bias)));
        }
      } break;
      case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error('Invalid parameters number of ATROUS_DEPTHWISE_CONV_2D');
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        const strides = [1, 1];
        let dilationW, dilationH;
        let depthMultipler;
        let activation;
        let paddedInput = input;
        const inputInChannels = input.shape[3];
        const filterInChannels = filter.shape[2];
        if (inputInChannels < filterInChannels) {
          paddedInput = input.pad([[0, 0], [0, 0], [0, 0], [0, filterInChannels - inputInChannels]]);
        }
        if (inCount === 8) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          dilationW = operands[inputs[i++]].value[0];
          dilationH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              paddedInput.depthwiseConv2D(filter, strides, padding, 'NHWC', [dilationH, dilationW]).add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          dilationW = operands[inputs[i++]].value[0];
          dilationH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              paddedInput.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                         .depthwiseConv2D(filter, strides, 'valid', 'NHWC', [dilationH, dilationW]).add(bias)));
        }
      } break;
      case OperationCode.AVERAGE_POOL_2D:
      case OperationCode.MAX_POOL_2D: {
        const inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error(`Invalid parameters number of Pooling ${op}`);
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const output = operands[outputs[0]];
        let strideW, strideH;
        let filterW, filterH;
        let activation;
        if (inCount === 7) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          filterW = operands[inputs[i++]].value[0];
          filterH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          if (op === OperationCode.AVERAGE_POOL_2D) {
            output.assign(activation(
                input.avgPool([filterH, filterW], [strideH, strideW], padding)));
          } else {
            output.assign(activation(
                input.maxPool([filterH, filterW], [strideH, strideW], padding)));
          }
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          filterW = operands[inputs[i++]].value[0];
          filterH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          if (op === OperationCode.AVERAGE_POOL_2D) {
            output.assign(activation(
                input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                     .avgPool([filterH, filterW], [strideH, strideW], 'valid')));
          } else {
            output.assign(activation(
                input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                     .maxPool([filterH, filterW], [strideH, strideW], 'valid')));
          }
        }
      } break;
      case OperationCode.SOFTMAX: {
        const input = operands[inputs[0]];
        const beta = operands[inputs[1]].value[0];
        const output = operands[outputs[0]];
        if (beta === 1) {
          output.assign(input.softmax());
        } else {
          output.assign(input.mul(tf.scalar(beta)).softmax());
        }
      } break;
      case OperationCode.RESHAPE: {
        const input = operands[inputs[0]];
        const targetShape = operands[inputs[1]];
        const output = operands[outputs[0]];
        if (targetShape.value === undefined) {
          targetShape.value = targetShape.dataSync();
        }
        output.assign(input.reshape(targetShape.value));
      } break;
      case OperationCode.CONCATENATION: {
        const numInputTensors = inputs.length - 1;
        const axis = operands[inputs[numInputTensors]].value[0];
        const output = operands[outputs[0]];
        let inputTensors = [];
        for (let i = 0; i < numInputTensors; ++i) {
          inputTensors.push(operands[inputs[i]]);
        }
        output.assign(tf.concat(inputTensors, axis));
      } break;
      case OperationCode.FULLY_CONNECTED: {
        const input = operands[inputs[0]];
        const weights = operands[inputs[1]];
        const bias = operands[inputs[2]];
        const activation = FuseFunctionMap.get(operands[inputs[3]].value[0]);
        const output = operands[outputs[0]];
        const batchSize = utils.product(input.shape) / weights.shape[1];
        output.assign(activation(
            tf.matMul(input.reshape([batchSize, -1]), weights, false, true).add(bias)));
      } break;
      default: {
        throw new Error(`Operation ${op} is not supported`);
      }
    }
  }

  /**
   * Types supported in tfjs: float32, int32, bool, complex64 
   */
  _getOperandType(type) {
    if (type === OperandCode.TENSOR_FLOAT32) {
      return 'float32';
    } else if (type === OperandCode.TENSOR_INT32) {
      return 'int32';
    } else {
      throw new Error(`Operand type ${type} is not supproted`);
    }
  }

  /**
   * Change (depthwise)conv2d weights format 
   */
  _changeWeightsFormat() {
    this._operations.forEach(operation => {
      const op = operation.type;
      switch(op) {
        case OperationCode.CONV_2D:
        case OperationCode.ATROUS_CONV_2D: {
          // NHWC -> HWCN
          // https://js.tensorflow.org/api/0.13.3/#conv2d
          const inputs = operation.inputs;
          const filter = this._operands[inputs[1]];
          this._operands[inputs[1]] = filter.transpose([1, 2, 3, 0]);
          filter.dispose();
        } break;
        case OperationCode.DEPTHWISE_CONV_2D:
        case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
          // [1, filterH, filterW, outChannels] -> [filterH, filterW, inChannels, depthMultipler]
          // https://js.tensorflow.org/api/0.13.3/#depthwiseConv2d
          const inputs = operation.inputs;
          const filter = this._operands[inputs[1]];
          const filterH = filter.shape[1];
          const filterW = filter.shape[2];
          const depthMultipler =  this._operands[inputs[inputs.length-2]].value[0];
          this._operands[inputs[1]] = filter.reshape([filterH, filterW, -1, depthMultipler]);
          filter.dispose();
        } break;
      }
    });
  }

  _deleteAll() {
    this._operands.forEach(operand => {
      if (operand.isDisposed === false) {
        operand.dispose();
      }
    })
  }

  static _supportWebGL() {
    return tf.getBackend() === 'webgl';
  }
}

