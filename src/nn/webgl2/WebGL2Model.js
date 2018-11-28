import * as utils from '../utils'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime} from '../Enums'
import * as tf from './tfjs-core/dist/index';

export default class WebGL2Model {
  /**
   * Create WebGL2Model class in nn/Model.js
   *
   * @param {Object} model - Model from nn/Model.js
   */
  constructor(model) {
    this._model = model;
    this._operations = model._operations;
    this._operands = [];
    this._prepared = false;

    // console.log(tf.ENV);
    // console.log(tf.ENV.backend.floatPrecision());
  }

  /**
   * Called in nn/Compilation.js
   *
   */
  prepareModel() {
    this._model._operands.forEach(operand => {
      if (utils.isTensor(operand.type)) {
        let type = this._getOperandType(operand.type);
        if (operand.value !== null) {   
          // constant tensor
          this._operands.push(tf.tensor(operand.value, operand.dimensions, type));
        } else {                        
          // variable tensor 
          let zeroTensor = tf.zeros(operand.dimensions, type);
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
      let operand = this._operands[input.index];
      let inputTensor = tf.tensor(input.buffer, operand.shape, operand.dtype);
      operand.assign(inputTensor);
      inputTensor.dispose();
    });

    this._operations.forEach(operation => {
      tf.tidy(() => {
        this._executeOperation(operation);
      });
    });

    outputs.forEach(output => {
      let operand = this._operands[output.index];  
      output.buffer.set(operand.dataSync());
    });
    
    // console.log(tf.memory());
  }

  _executeOperation(operation) {
    let op = operation.type;
    let inputs = operation.inputs;
    let outputs = operation.outputs;
    let operands = this._operands;

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
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseFunctionMap.get(operands[inputs[2]].value[0]);
        let output = operands[outputs[0]];
        output.assign(activation(tf.add(in1, in2)));
      } break;
      case OperationCode.MUL: {
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = FuseFunctionMap.get(operands[inputs[2]].value[0]);
        let output = operands[outputs[0]];
        output.assign(activation(tf.mul(in1, in2)));
      } break;
      case OperationCode.CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of CONV_2D');
        }
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let output = operands[outputs[0]];
        let strideW, strideH;
        let activation;
        if (inCount === 7) {
          let paddingCode = operands[inputs[i++]].value[0];
          let padding = PaddingCodeMap.get(paddingCode);
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
            input.conv2d(filter, [strideH, strideW], padding).add(bias)));
        } else {
          let paddingLeft = operands[inputs[i++]].value[0];
          let paddingRight = operands[inputs[i++]].value[0];
          let paddingTop = operands[inputs[i++]].value[0];
          let paddingBottom = operands[inputs[i++]].value[0];
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
            input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                 .conv2d(filter, [strideH, strideW], 'valid').add(bias)));
        }
      } break;
      case OperationCode.DEPTHWISE_CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error('Invalid parameters number of DEPTHWISE_CONV_2D');
        }
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let output = operands[outputs[0]];
        let strideW, strideH;
        let depthMultipler;
        let activation;
        if (inCount === 8) {
          let paddingCode = operands[inputs[i++]].value[0];
          let padding = PaddingCodeMap.get(paddingCode);
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
            input.depthwiseConv2D(filter, [strideH, strideW], padding).add(bias)));
        } else {
          let paddingLeft = operands[inputs[i++]].value[0];
          let paddingRight = operands[inputs[i++]].value[0];
          let paddingTop = operands[inputs[i++]].value[0];
          let paddingBottom = operands[inputs[i++]].value[0];
          strideW = operands[inputs[i++]].value[0];
          strideH = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
            input.pad([[0, 0], [paddingTop, paddingBottom], [paddingLeft, paddingRight], [0, 0]])
                 .depthwiseConv2D(filter, [strideH, strideW], 'valid').add(bias)));
        }
      } break;
      case OperationCode.AVERAGE_POOL_2D:
      case OperationCode.MAX_POOL_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error(`Invalid parameters number of Pooling ${op}`);
        }
        let i = 0;
        let input = operands[inputs[i++]];
        let output = operands[outputs[0]];
        let strideW, strideH;
        let filterW, filterH;
        let activation;
        if (inCount === 7) {
          let paddingCode = operands[inputs[i++]].value[0];
          let padding = PaddingCodeMap.get(paddingCode);
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
          let paddingLeft = operands[inputs[i++]].value[0];
          let paddingRight = operands[inputs[i++]].value[0];
          let paddingTop = operands[inputs[i++]].value[0];
          let paddingBottom = operands[inputs[i++]].value[0];
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
        let input = operands[inputs[0]];
        let beta = operands[inputs[1]].value[0];
        let output = operands[outputs[0]];
        output.assign(input.mul(tf.scalar(beta)).softmax());
      } break;
      case OperationCode.RESHAPE: {
        let input = operands[inputs[0]];
        let targetShape = operands[inputs[1]];
        let output = operands[outputs[0]];
        output.assign(input.reshape(targetShape.dataSync()));
      } break;
      case OperationCode.CONCATENATION: {
        if (outputs.length < 1 || inputs.length < 2) {
          throw new Error('Invalid inputs or outputs');
        }
        let numInputTensors = inputs.length - 1;
        let axis = operands[inputs[numInputTensors]].value[0];
        let output = operands[outputs[0]];
        let inputTensors = [];
        for (let i = 0; i < numInputTensors; ++i) {
          inputTensors.push(operands[inputs[i]]);
        }
        output.assign(tf.concat(inputTensors, axis));
      } break;
      case OperationCode.FULLY_CONNECTED: {
        let input = operands[inputs[0]];
        let weights = operands[inputs[1]];
        let bias = operands[inputs[2]];
        let activation = FuseFunctionMap.get(operands[inputs[3]].value[0]);
        let output = operands[outputs[0]];
        let batchSize = input.shape[0];
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
      let op = operation.type;
      switch(op) {
        case OperationCode.CONV_2D: {
          // NHWC -> HWCN
          let inputs = operation.inputs;
          let filter = this._operands[inputs[1]];
          this._operands[inputs[1]] = filter.transpose([1, 2, 3, 0]);
          filter.dispose();
        } break;
        case OperationCode.DEPTHWISE_CONV_2D: {
          // [1, filterH, filterW, outChannels] -> [filterH, filterW, inChannels, depthMultipler]
          let inputs = operation.inputs;
          let input = this._operands[inputs[0]];
          let filter = this._operands[inputs[1]];
          let filterH = filter.shape[1];
          let filterW = filter.shape[2];
          let inChannels = input.shape[3];
          let depthMultipler =  this._operands[inputs[inputs.length-2]].value[0];
          this._operands[inputs[1]] = filter.reshape([filterH, filterW, inChannels, depthMultipler]);
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

  static _supportWebGL2() {
    return tf.getBackend() === 'webgl' && tf.ENV.get('WEBGL_VERSION') === 2;
  }
}

