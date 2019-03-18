import {OperationCode, OperandCode, PaddingCode, FuseCode} from '../Enums'
import * as utils from '../utils'
import * as tf from '@tensorflow/tfjs-core';
import Graph from '../GraphUtils';

var executeTimes = 0;
var skipWarmUpRuns = 1;
var profiling = [];

export default class WebGLModel {
  /**
   * Create WebGLModel class in nn/Model.js
   *
   * @param {Object} model - Model from nn/Model.js
   */
  constructor(model) {
    this._nnNative = navigator.ml.getNeuralNetworkContext();
    this._supportedOpCode = new Set([]);
    this._model = model;
    this._operations = [];
    this._operands = [];
    this._nnOperands = [];
    this._syncedFromWebGL = [];
    this._prepared = false;

    if (tf.ENV.backend.floatPrecision() === 16) {
      console.warn(
          'The current floating point operation precision is only 16-bit');
    }
  }

  /** Called in nn/Compilation.js */
  async prepareModel() {
    const model = this._model;

    this._hybridPreferCode = {
      low: this._nnNative.PREFER_LOW_POWER,
      fast: this._nnNative.PREFER_FAST_SINGLE_ANSWER,
      sustained: this._nnNative.PREFER_SUSTAINED_SPEED,
    }[model.hybridPrefer];
    console.debug(`Backend: WebGL + ${model.hybridPrefer}`);
    this._supportedOpCode = new Set(model.supportedOpsList);
    console.debug(`Supported Ops: ${Array.from(this._supportedOpCode).map(op => Object.keys(OperationCode).find(k => OperationCode[k] === op)).join(', ') || 'None'}`);

    const graph = new Graph(model._operations.length);
    model._operations.forEach((op, i) => {
      graph.addNode(i, op.inputs, op.outputs);
      if (!this._supportedOpCode.has(op.type)) {
        graph.setBlack(i);
      }
    })
    graph.identifyInputOutputTensors(model._inputs, model._outputs);

    let isEagerMode = model.eagerMode;
    console.debug(`Mode: ${isEagerMode ? 'Eager' : 'Graph'}`);

    const partitions = graph.partition(isEagerMode);

    for (const {nodes, inTensors, outTensors} of partitions) {
      const subgraphName = `Subgraph ${typeof this.subgraphcounter === 'undefined' ? this.subgraphcounter = 0 : ++this.subgraphcounter}\t (${this._supportedOpCode.has(model._operations[nodes[0]].type) ? 'WebNN' : 'WebGL'}):\t{${Object.entries(nodes.map(opId => Object.keys(OperationCode).find(k => OperationCode[k] === model._operations[opId].type)).reduce((counts, v) => {counts[v]?counts[v]++:counts[v]=1; return counts}, {})).map(n => `${n[0]} x ${n[1]}`).join(', ')}}`;
      console.debug(subgraphName);

      if (!this._supportedOpCode.has(model._operations[nodes[0]].type)) {

        // run in polyfil

        // break group of WebGL operaions to singletons in eager mode
        for (const operationId of nodes) {
          const operation = model._operations[operationId];
          operation.subgraphName = subgraphName;
          this._operations.push(operation);

          // allocate WebGL runtime textures

          for (const tensorId of [...operation.inputs, ...operation.outputs]) {
            const operand = this._model._operands[tensorId];
            if (utils.isTensor(operand.type)) {
              const type = this._getOperandType(operand.type);
              if (operand.value !== null) {   
                // constant tensor
                this._operands[tensorId] =
                    tf.tensor(operand.value, operand.dimensions, type);
              } else {                        
                // variable tensor 
                const zeroTensor = tf.zeros(operand.dimensions, type);
                this._operands[tensorId] = tf.variable(zeroTensor);
                zeroTensor.dispose();
              }
            } else {
              this._operands[tensorId] = operand;   
            }
          }
        }

      } else {

        // run in WebNN

        // allocate placeholders for WebNN operands copies
        for (const tensorId of [...inTensors, ...outTensors]) {
          if (!this._nnOperands.hasOwnProperty(tensorId)) {
            const tensor = model._operands[tensorId];
            const typedArray = utils.operandCodeToTypedArrayMap.get(tensor.type);
            this._nnOperands[tensorId] = new typedArray(utils.product(tensor.dimensions));
          }
        }

        // build subgraph
        const submodel = await this._nnNative.createModel();
        const globalIdToLocalId = {};
        let operandIndex = 0;

        for (const operationId of nodes) {
          const operation = model._operations[operationId];
          for (const tensorId of [...operation.inputs, ...operation.outputs]) {
            const globalTensorId = parseInt(tensorId);
            if (!globalIdToLocalId.hasOwnProperty(globalTensorId)) {
              const localTensorId = operandIndex++;
              globalIdToLocalId[globalTensorId] = localTensorId;
              const operand = model._operands[globalTensorId];
              const operandType = {
                type: operand.type,
                dimensions: operand.dimensions,
                scale: operand.scale,
                zeroPoint: operand.operand,
              };
              submodel.addOperand(operandType);
              if (operand.value) {
                submodel.setOperandValue(localTensorId, operand.value);
              }
            }
          }

          const operationInputs = operation.inputs.map(i => globalIdToLocalId[i]);
          const operationOutputs = operation.outputs.map(i => globalIdToLocalId[i]);
          submodel.addOperation(operation.type, operationInputs, operationOutputs);
        }

        const submodelInputs = inTensors.map(i => globalIdToLocalId[i]);
        const submodelOutputs = outTensors.map(i => globalIdToLocalId[i]);
        submodel.identifyInputsAndOutputs(submodelInputs, submodelOutputs);
        await submodel.finish();

        const compilation = await submodel.createCompilation();
        compilation.setPreference(this._hybridPreferCode);
        await compilation.finish();

        const execution = await compilation.createExecution();
        outTensors.forEach((tensorId, i) => {
          execution.setOutput(i, this._nnOperands[tensorId]);
        });

        this._operations.push({
          type: OperationCode.NATIVE_OP,
          inputs: inTensors,
          outputs: outTensors,
          execution: execution,
          subgraphName: subgraphName,
        });
      }

    }

    profiling = new Array(this._operations.length).fill(0);
    this._changeWeightsFormat();
    this._prepared = true;
  }

  /**
   * Called in nn/Execution.js
   *
   * @param {Map} inputs 
   * @param {Map} outputs 
   */
  async execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }

    executeTimes++;
    if (executeTimes === skipWarmUpRuns) {
      profiling.fill(0);
    }

    // wire up WebNN input tensors
    inputs.forEach((input) => {
      this._nnOperands[input.index] = input.buffer;
    });

    let start = performance.now();

    const firstOp = this._operations[0];
    if (firstOp.type !== OperationCode.NATIVE_OP) {
      // copy to WebGL texture
      inputs.forEach(input => {
        const operand = this._operands[input.index];
        const inputTensor =
            tf.tensor(input.buffer, operand.shape, operand.dtype);
        operand.assign(inputTensor);
        inputTensor.dispose();
      });
    }

    for (const [i, operation] of this._operations.entries()) {

      if (operation.type === OperationCode.NATIVE_OP) {
        let end = performance.now();
        profiling[i - 1] += end - start;
        start = end;

        await this._executeNNOperation(operation);

        end = performance.now();
        profiling[i] += end - start;
        start = end;
      } else {
        tf.tidy(() => this._executeGlOperation(operation));
      }

    }

    const lastOp = this._operations[this._operations.length - 1];
    if (lastOp.type !== OperationCode.NATIVE_OP) {
      // copy from WebGL texture
      outputs.forEach(output => {
        const operand = this._operands[output.index];  
        output.buffer.set(operand.dataSync());
      });

      const end = performance.now();
      profiling[this._operations.length - 1] += end - start;
    } else {
      outputs.forEach((output) => {
        const operand = this._nnOperands[output.index];  
        output.buffer.set(operand);
      });
    }
  }

  async _executeNNOperation(operation) {
    const inputs = operation.inputs;
    const outputs = operation.outputs;
    const execution = operation.execution;
    const nnOperands = this._nnOperands;
    const glOperands = this._operands;

    inputs.forEach((tensorId, i) => {
      const buffer = nnOperands[tensorId];
      execution.setInput(i, buffer);
    });

    // execute subgraph
    await execution.startCompute();

    outputs.forEach(tensorId => {
      // sync data to webgl if needed
      if (glOperands.hasOwnProperty(tensorId)) {
        const buffer = nnOperands[tensorId];
        const operand = glOperands[tensorId];
        const tmpTensor = tf.tensor(buffer, operand.shape, operand.dtype);
        operand.assign(tmpTensor);
        tmpTensor.dispose();
      }
    });
  }

  _executeGlOperation(operation) {
    const op = operation.type;
    const inputs = operation.inputs;
    const outputs = operation.outputs;
    const operands = this._operands;
    const nnOperands = this._nnOperands;

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
      case OperationCode.ADD:
      case OperationCode.MUL: {
        const input1 = operands[inputs[0]];
        const input2 = operands[inputs[1]];
        const activation = FuseFunctionMap.get(operands[inputs[2]].value[0]);
        const output = operands[outputs[0]];
        if (op === OperationCode.ADD) {
          output.assign(activation(tf.add(input1, input2)));
        } else {
          output.assign(activation(tf.mul(input1, input2)));
        }
      } break;
      case OperationCode.CONV_2D:
      case OperationCode.ATROUS_CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error(`Invalid parameters number of Conv2d ${op}`);
        }
        let i = 0;
        const input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        let strideW, strideH;
        let dilationW, dilationH;
        let activation;
        if (inCount === 7) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          if (op === OperationCode.CONV_2D) {
            strideW = operands[inputs[i++]].value[0];
            strideH = operands[inputs[i++]].value[0];
            [dilationW, dilationH] = [1, 1];
          } else {
            dilationW = operands[inputs[i++]].value[0];
            dilationH = operands[inputs[i++]].value[0];
            [strideW, strideH] = [1, 1];
          }
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.conv2d(filter, [strideH, strideW],
                           padding, 'NHWC',
                           [dilationH, dilationW])
                   .add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.CONV_2D) {
            strideW = operands[inputs[i++]].value[0];
            strideH = operands[inputs[i++]].value[0];
            [dilationW, dilationH] = [1, 1];
          } else {
            dilationW = operands[inputs[i++]].value[0];
            dilationH = operands[inputs[i++]].value[0];
            [strideW, strideH] = [1, 1];
          }
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          if (this._isPaddingEqual(paddingLeft, paddingRight,
                                   paddingTop, paddingBottom)) {
            output.assign(activation(
                input.conv2d(filter, [strideH, strideW],
                             paddingLeft, 'NHWC',
                             [dilationH, dilationW], 'floor')
                     .add(bias)));
          } else {
            output.assign(activation(
                input.pad([[0, 0], [paddingTop, paddingBottom],
                           [paddingLeft, paddingRight], [0, 0]])
                     .conv2d(filter, [strideH, strideW],
                             'valid', 'NHWC',
                             [dilationH, dilationW])
                     .add(bias)));
          }
        }
      } break;
      case OperationCode.DEPTHWISE_CONV_2D:
      case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
        const inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error(
              `Invalid parameters number of DepthwiseConv2d ${op}`);
        }
        let i = 0;
        let input = operands[inputs[i++]];
        const filter = operands[inputs[i++]];
        const bias = operands[inputs[i++]];
        const output = operands[outputs[0]];
        let strideW, strideH;
        let dilationW, dilationH;
        let depthMultipler;
        let activation;
        // pad input if inputChannels is less than filterChannels
        const inputChannels = input.shape[3];
        const filterChannels = filter.shape[2];
        if (inputChannels < filterChannels) {
          input = input.pad([[0, 0], [0, 0],
                             [0, 0], [0, filterChannels - inputChannels]]);
        }
        if (inCount === 8) {
          const paddingCode = operands[inputs[i++]].value[0];
          const padding = PaddingCodeMap.get(paddingCode);
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideW = operands[inputs[i++]].value[0];
            strideH = operands[inputs[i++]].value[0];
            [dilationW, dilationH] = [1, 1];
          } else {
            dilationW = operands[inputs[i++]].value[0];
            dilationH = operands[inputs[i++]].value[0];
            [strideW, strideH] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          output.assign(activation(
              input.depthwiseConv2D(filter, [strideH, strideW],
                                    padding, 'NHWC',
                                    [dilationH, dilationW])
                   .add(bias)));
        } else {
          const paddingLeft = operands[inputs[i++]].value[0];
          const paddingRight = operands[inputs[i++]].value[0];
          const paddingTop = operands[inputs[i++]].value[0];
          const paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideW = operands[inputs[i++]].value[0];
            strideH = operands[inputs[i++]].value[0];
            [dilationW, dilationH] = [1, 1];
          } else {
            dilationW = operands[inputs[i++]].value[0];
            dilationH = operands[inputs[i++]].value[0];
            [strideW, strideH] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = FuseFunctionMap.get(operands[inputs[i++]].value[0]);
          if (this._isPaddingEqual(paddingLeft, paddingRight,
                                   paddingTop, paddingBottom)) {
            output.assign(activation(
                input.depthwiseConv2D(filter, [strideH, strideW],
                                      paddingLeft, 'NHWC',
                                      [dilationH, dilationW], 'floor')
                     .add(bias)));
          } else {
            output.assign(activation(
                input.pad([[0, 0], [paddingTop, paddingBottom],
                           [paddingLeft, paddingRight], [0, 0]])
                     .depthwiseConv2D(filter, [strideH, strideW],
                                      'valid', 'NHWC',
                                      [dilationH, dilationW])
                     .add(bias)));
          }
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
                input.avgPool([filterH, filterW],
                              [strideH, strideW],
                              padding)));
          } else {
            output.assign(activation(
                input.maxPool([filterH, filterW],
                              [strideH, strideW],
                              padding)));
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
          if (this._isPaddingEqual(paddingLeft, paddingRight,
                                   paddingTop, paddingBottom)) {
            if (op === OperationCode.AVERAGE_POOL_2D) {
              output.assign(activation(
                  input.avgPool([filterH, filterW],
                                [strideH, strideW],
                                paddingLeft, 'floor')));
            } else {
              output.assign(activation(
                  input.maxPool([filterH, filterW],
                                [strideH, strideW],
                                paddingLeft, 'floor')));
            }            
          } else {
            if (op === OperationCode.AVERAGE_POOL_2D) {
              throw new Error(
                  'AVERAGE_POOL_2D with unequal padding is not supported');
            } else {
              output.assign(activation(
                  input.pad([[0, 0], [paddingTop, paddingBottom],
                             [paddingLeft, paddingRight], [0, 0]],
                            -1e8 /* a small enough constant */)
                       .maxPool([filterH, filterW],
                                [strideH, strideW],
                                'valid')));
            }
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
            input.reshape([batchSize, -1])
                 .matMul(weights, false, true)
                 .add(bias)));
      } break;
      case OperationCode.RESIZE_BILINEAR: {
        if (outputs.length < 1 || inputs.length < 3) {
          throw new Error('Invalid inputs or outputs');
        }
        const input = operands[inputs[0]];
        const newHeight = operands[inputs[1]].value[0];
        const newWidth = operands[inputs[2]].value[0];
        const output = operands[outputs[0]];
        let alignCorner = false;
        if (inputs.length === 4) {
          alignCorner = operands[inputs[3]].value[0] !== 0;
        }
        output.assign(
            input.resizeBilinear([newHeight, newWidth], alignCorner));
      } break;
      case OperationCode.TANH: {
        const input = operands[inputs[0]];
        const output = operands[outputs[0]];
        output.assign(input.tanh());
      } break;
      case OperationCode.BATCH_TO_SPACE_ND: {
        const input = operands[inputs[0]];
        const blockShape = operands[inputs[1]];
        const output = operands[outputs[0]];
        const crops = [[0, 0], [0, 0]];
        if (blockShape.value === undefined) {
          // blockShape.dataSync() return Int32Array,
          // which should be converted to Array here.
          blockShape.value = Array.apply([], blockShape.dataSync());
        }
        output.assign(input.batchToSpaceND(blockShape.value, crops));
      } break;
      case OperationCode.TRANSPOSE: {
        const input = operands[inputs[0]];
        const perm = operands[inputs[1]];
        const output = operands[outputs[0]];
        if (perm !== undefined) {
          if (perm.value === undefined) {
            perm.value = perm.dataSync();
          }
          output.assign(input.transpose(perm.value));
        } else {
          output.assign(input.transpose());
        }
      } break;
      case OperationCode.MAXIMUM: {
        const input1 = operands[inputs[0]];
        const input2 = operands[inputs[1]];
        const output = operands[outputs[0]];
        output.assign(tf.maximum(input1, input2));
      } break;
      default: {
        throw new Error(`Operation ${op} is not supported`);
      }
    }

    outputs.forEach(tensorId => {
      // sync data to webnn if needed
      if (nnOperands.hasOwnProperty(tensorId)) {
        const buffer = nnOperands[tensorId];
        const operand = operands[tensorId];
        buffer.set(operand.dataSync());
      }
    });
  }

  /** Types supported in tfjs: float32, int32, bool, complex64 */
  _getOperandType(type) {
    if (type === OperandCode.TENSOR_FLOAT32) {
      return 'float32';
    } else if (type === OperandCode.TENSOR_INT32) {
      return 'int32';
    } else {
      throw new Error(`Operand type ${type} is not supproted`);
    }
  }

  /** Change (depthwise) conv2d weights format. */
  _changeWeightsFormat() {
    this._operations.forEach(operation => {
      const op = operation.type;
      switch(op) {
        case OperationCode.CONV_2D:
        case OperationCode.ATROUS_CONV_2D: {
          // [outChannels, filterH, filterW, inChannels]
          // => [filterH, filterW, inChannels, outChannels]
          // https://js.tensorflow.org/api/0.14.1/#conv2d
          const inputs = operation.inputs;
          const filter = this._operands[inputs[1]];
          this._operands[inputs[1]] = filter.transpose([1, 2, 3, 0]);
          filter.dispose();
        } break;
        case OperationCode.DEPTHWISE_CONV_2D:
        case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
          // [1, filterH, filterW, outChannels]
          // => [filterH, filterW, inChannels, depthMultipler]
          // https://js.tensorflow.org/api/0.14.1/#depthwiseConv2d
          const inputs = operation.inputs;
          const filter = this._operands[inputs[1]];
          const filterH = filter.shape[1];
          const filterW = filter.shape[2];
          const depthMultipler =
              this._operands[inputs[inputs.length - 2]].value[0];
          this._operands[inputs[1]] =
              filter.reshape([filterH, filterW, -1, depthMultipler]);
          filter.dispose();
        } break;
      }
    });
  }

  _isPaddingEqual(left, right, top, bottom) {
    return (left === right) && (left === top) && (left === bottom);
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

  getReport() {
    executeTimes -= skipWarmUpRuns;
    let webglTime = 0;
    let webnnTime = 0;
    console.debug(`\n\nExecution calls: ${executeTimes} (omitted ${skipWarmUpRuns} warm-up runs)`);
    console.debug(`Note: Sync time is included in WebGL op.`);
    for (const [i, op] of this._operations.entries()) {
      const avgTime = profiling[i] / executeTimes;
      if (!avgTime) {
        continue;
      }
      console.debug(`${avgTime.toFixed(5)} ms\t- ${op.subgraphName}`);
      if (op.subgraphName.indexOf('WebGL') > 0) {
        webglTime += avgTime;
      } else {
        webnnTime += avgTime;
      }
    }
    console.debug(`WebGL time: ${webglTime.toFixed(5)} ms`);
    console.debug(`WebNN time: ${webnnTime.toFixed(5)} ms`);
    console.debug(`Sum: ${(webglTime + webnnTime).toFixed(5)} ms`);
    executeTimes = 0;
  }
}

