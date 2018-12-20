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
    this._model;
    this._toDelete = {
      tensorValue: [],
      tensorShape: []
    };
  }

  /**
   * Prepare for model execution.
   * 
   * @param {Object} model - A model object built by user.
   */
  async prepare(model) {
    this._model = model;
    this._nn_ops = await getNNOpsInstance();
    this._operations = model._operations;
    for (let i = 0; i < model._operands.length; ++i) {
      let operand = model._operands[i];
      let runtimeOperand = {};
      runtimeOperand.type = operand.type;
      if (utils.isTensor(operand.type)) {
        runtimeOperand.value = this._allocateTensor(operand);
        runtimeOperand.runtimeshape = this._allocateRuntimeShape(operand);
        this._toDelete.tensorValue.push(runtimeOperand.value);
        this._toDelete.tensorShape.push(runtimeOperand.runtimeshape);
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

    function calculateExplicitPadding(inSize, stride, filterSize, paddingCode) {
      let paddingHead = 0;
      let paddingTail = 0;

      if (paddingCode === PaddingCode.SAME) {
        let outSize = Math.floor((inSize + stride - 1) / stride);
        let tmp = Math.floor((outSize - 1) * stride + filterSize);
        if (tmp > inSize) {
          paddingHead = Math.floor((tmp - inSize) / 2);
          paddingTail = Math.floor((tmp - inSize) - paddingHead);
        }
      }

      return [paddingHead, paddingTail];
    }

    function calculateActivationRangeFloat(activation) {
      let activation_min, activation_max;
      if (activation === FuseCode.RELU) {
        activation_min = 0.0;
        activation_max = nn_ops.MAX;
      } else if (activation === FuseCode.RELU6) {
        activation_min = 0.0;
        activation_max = 6.0;
      } else if (activation === FuseCode.RELU1) {
        activation_min = -1.0;
        activation_max = 1.0;
      } else if (activation === FuseCode.NONE){
        activation_min = nn_ops.LOWEST;
        activation_max = nn_ops.MAX;
      } else {
        throw new Error("Unsupported fused activation function.");
      }
      return {activation_min, activation_max};
    }

    function sameShape(input1, input2) {
      if (input1.type != input2.type || input1.runtimeshape.size != input2.runtimeshape.size){
          return false;
      }
      for (let i = 0; i < input1.runtimeshape.size; i++) {
          if (input1.runtimeshape.dims[i] != input2.runtimeshape.dims[i]) {
              return false;
          }
      }
      return true;
    }  

    function OPS_CHECK(option) {
      if (!option) {
        throw new Error(`OPS_CHECK failed`);
      }
      return true;
    }

    switch(op) {
      case OperationCode.ADD: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = operands[inputs[2]].value[0];
        let out = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(in1.type === in2.type);
        OPS_CHECK(in1.runtimeshape.size <= 4 && in2.runtimeshape.size <= 4);

        // init arithmeticParams
        let arithmeticParams = new nn_ops.ArithmeticParams;
        arithmeticParams.float_activation_range = [float_activation_min, float_activation_max];
        
        let needBroadCast = !sameShape(in1, in2);
        let funcName;
        if (needBroadCast) {
          success = nn_ops.broadCastAddFloat32(arithmeticParams,
                                               in1.runtimeshape, in1.value,
                                               in2.runtimeshape, in2.value,
                                               out.runtimeshape, out.value);
          funcName = `broadCastAddFloat32`;
        } else {
          success = nn_ops.addFloat32(arithmeticParams,
                                      in1.runtimeshape, in1.value,
                                      in2.runtimeshape, in2.value,
                                      out.runtimeshape, out.value);
          funcName = `addFloat32`;
        }
        if (!success) {
          throw new Error(`${funcName} fails`);
        }
        arithmeticParams.delete();
      } break;
      case OperationCode.MUL: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = operands[inputs[2]].value[0];
        let out = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(in1.type === in2.type);
        OPS_CHECK(in1.runtimeshape.size <= 4 && in2.runtimeshape.size <= 4);

        // init arithmeticParams
        let arithmeticParams = new nn_ops.ArithmeticParams;
        arithmeticParams.float_activation_range = [float_activation_min, float_activation_max];

        let needBroadCast = !sameShape(in1, in2);
        let funcName;
        if (needBroadCast) {
          success = nn_ops.broadCastMulFloat32(arithmeticParams,
                                               in1.runtimeshape, in1.value,
                                               in2.runtimeshape, in2.value,
                                               out.runtimeshape, out.value);
          funcName = `broadCastMulFloat32`;
        } else {
          success = nn_ops.mulFloat32(arithmeticParams,
                                      in1.runtimeshape, in1.value,
                                      in2.runtimeshape, in2.value,
                                      out.runtimeshape, out.value);
          funcName = `mulFloat32`;
        }
        if (!success) {
          throw new Error(`${funcName} fails`);
        }
        arithmeticParams.delete();
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
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let filterWidth = filter.runtimeshape.dims[2];
        let filterHeight = filter.runtimeshape.dims[1];
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.dims[2];
          let inputHeight = input.runtimeshape.dims[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        // init im2col operand
        let outBatch = output.runtimeshape.dims[0];
        let outHeight = output.runtimeshape.dims[1];
        let outWidth = output.runtimeshape.dims[2];
        let inDepth = input.runtimeshape.dims[3];
        let im2colDepth = filterWidth * filterHeight * inDepth;
        let im2colDims = [outBatch, outHeight, outWidth, im2colDepth];
        let im2colValue = new Float32Array(utils.product(im2colDims));
        let operand = {
          type: OperandCode.TENSOR_FLOAT32,
          dimensions: im2colDims,
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: im2colValue
        }
        let im2colShape = this._allocateRuntimeShape(operand);
        let im2colData = this._allocateTensor(operand);

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(input.type === filter.type);
        if (input.type === OperandCode.TENSOR_QUANT8_ASYMM) {
            OPS_CHECK(bias.type === OperandCode.TENSOR_INT32);
        } else {
            OPS_CHECK(input.type === bias.type);
        }

        OPS_CHECK(input.runtimeshape.size === 4);
        OPS_CHECK(filter.runtimeshape.size === 4);
        OPS_CHECK(bias.runtimeshape.size === 1);
        OPS_CHECK(output.runtimeshape.size === 4);

        OPS_CHECK(filter.runtimeshape.dims[0] === bias.runtimeshape.dims[0]);
        OPS_CHECK(filter.runtimeshape.dims[3] === input.runtimeshape.dims[3]);

        // init convParams
        let convParams = new nn_ops.ConvParams;
        convParams.padding_values = [paddingLeft, paddingTop];
        convParams.strides = [strideWidth, strideHeight];
        convParams.dilation_factors = [1, 1];
        convParams.float_activation_range = [float_activation_min, float_activation_max];

        success = nn_ops.convFloat32(convParams, 
                                     input.runtimeshape, input.value, 
                                     filter.runtimeshape, filter.value, 
                                     bias.runtimeshape, bias.value, 
                                     output.runtimeshape, output.value,
                                     im2colShape, im2colData);
        if (!success) {
          throw new Error('convFloat32 fails');
        }
        convParams.delete();
        im2colShape.delete();
        nn_ops._free(im2colData);
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
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let depthMultipler;
        let activation;
        if (inCount === 11) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.dims[2];
          let inputHeight = input.runtimeshape.dims[1];
          let filterWidth = filter.runtimeshape.dims[2];
          let filterHeight = filter.runtimeshape.dims[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(input.type === filter.type);
        if (input.type === OperandCode.TENSOR_QUANT8_ASYMM) {
            OPS_CHECK(bias.type === OperandCode.TENSOR_INT32);
        } else {
            OPS_CHECK(input.type === bias.type);
        }

        OPS_CHECK(input.runtimeshape.size === 4);
        OPS_CHECK(filter.runtimeshape.size === 4);
        OPS_CHECK(bias.runtimeshape.size === 1);
        OPS_CHECK(output.runtimeshape.size === 4);

        OPS_CHECK(filter.runtimeshape.dims[3] === bias.runtimeshape.dims[0]);

        // init depthwiseParams
        let depthwiseParams = new nn_ops.DepthwiseParams;
        depthwiseParams.padding_values = [paddingLeft, paddingTop];
        depthwiseParams.strides = [strideWidth, strideHeight];
        depthwiseParams.dilation_factors = [1, 1];
        depthwiseParams.float_activation_range = [float_activation_min, float_activation_max];
        depthwiseParams.depth_multiplier = depthMultipler;

        success = nn_ops.depthwiseConvFloat32(depthwiseParams, 
                                              input.runtimeshape, input.value, 
                                              filter.runtimeshape, filter.value, 
                                              bias.runtimeshape, bias.value, 
                                              output.runtimeshape, output.value);
        if (!success) {
          throw new Error('depthwiseConvFloat32 fails');
        }
        depthwiseParams.delete();
      } break;
      case OperationCode.AVERAGE_POOL_2D:
      case OperationCode.MAX_POOL_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error(`Invalid parameters number of Pooling ${op}`);
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let filterWidth, filterHeight;
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          filterWidth = operands[inputs[i++]].value[0];
          filterHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          filterWidth = operands[inputs[i++]].value[0];
          filterHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.dims[2];
          let inputHeight = input.runtimeshape.dims[1];
          [paddingLeft, paddingRight] = calculateExplicitPadding(inputWidth, strideWidth, filterWidth, paddingCode);
          [paddingTop, paddingBottom] = calculateExplicitPadding(inputHeight, strideHeight, filterHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(input.runtimeshape.size === 4);
        OPS_CHECK(output.runtimeshape.size === 4);

        // init poolParams
        let poolParams = new nn_ops.PoolParams;
        poolParams.padding_values = [paddingLeft, paddingTop];
        poolParams.strides = [strideWidth, strideHeight];
        poolParams.filters = [filterWidth, filterHeight];
        poolParams.float_activation_range = [float_activation_min, float_activation_max];

        if (op === OperationCode.AVERAGE_POOL_2D) {
          success = nn_ops.averagePoolFloat32(poolParams, 
                                              input.runtimeshape, input.value,
                                              output.runtimeshape, output.value);
        } else if (op === OperationCode.MAX_POOL_2D) {
          success = nn_ops.maxPoolFloat32(poolParams, 
                                          input.runtimeshape, input.value,
                                          output.runtimeshape, output.value);
        }
        if (!success) {
          throw new Error(`Pooling ${op} fails`);
        }
        poolParams.delete();
      } break;
      case OperationCode.SOFTMAX: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let beta = operands[inputs[1]].value[0];
        if (beta <= 0.0) {
          throw new Error('beta must be positive for SOFTMAX');
        }
        let output = operands[outputs[0]];

        // Error check
        OPS_CHECK(input.runtimeshape.size <= 4);

        // init softmaxParams
        let softmaxParams = new nn_ops.SoftmaxParams;
        softmaxParams.beta = beta;

        success = nn_ops.softmaxFloat32(softmaxParams, 
                                        input.runtimeshape, input.value, 
                                        output.runtimeshape, output.value);
        if (!success) {
          throw new Error('softmaxFloat32 fails');
        }
        softmaxParams.delete();
      } break;
      case OperationCode.RESHAPE: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let targetShape = operands[inputs[1]];  // Dont use targetShape since outputShape has been set at first

        let output = operands[outputs[0]];

        let size_count;
        if (utils.isTensor(input.type)) {
          size_count = utils.sizeOfTensorData(input.type, input.runtimeshape.dims);
        } else {
          size_count = utils.sizeOfScalarData(input.type);
        }

        // Error check
        let numInputElements = utils.product(input.runtimeshape.dims);
        let numOutputElements = utils.product(output.runtimeshape.dims);
        OPS_CHECK(numInputElements === numOutputElements);

        // init reshapeParams
        let reshapeParams = new nn_ops.ReshapeParams;
        reshapeParams.size_count = size_count;

        success = nn_ops.reshapeGeneric(reshapeParams,
                                        input.runtimeshape, input.value,  
                                        output.runtimeshape, output.value);
        if (!success) {
          throw new Error('reshapeGeneric fails');
        }
        reshapeParams.delete();
      } break;
      case OperationCode.CONCATENATION: {
        if (outputs.length < 1 || inputs.length < 2) {
          throw new Error('Invalid inputs or outputs');
        }
        let numInputTensors = inputs.length - 1;
        let axis = operands[inputs[numInputTensors]].value[0];
        let input0 = operands[inputs[0]];
        let num_dimensions = input0.runtimeshape.size;
        let input_type = input0.type;
        if (axis === -1) {
          axis = num_dimensions - 1;
        }
        let output = operands[outputs[0]];
        let inputShapes = new nn_ops.VectorShape;
        let inputValues = new nn_ops.VectorPtr;
        for (let i = 0; i < numInputTensors; ++i) {
          let input = operands[inputs[i]];
          inputShapes.push_back(input.runtimeshape);
          inputValues.push_back(input.value);
        }

        // Error check
        OPS_CHECK(axis >= 0 && axis < num_dimensions);
        for (let  i = 1; i < numInputTensors; ++i) {
          let input = operands[inputs[i]];
          OPS_CHECK(input.runtimeshape.size === num_dimensions);
          OPS_CHECK(input.type === input_type);
          for (let d = 0; d < num_dimensions; ++d) {
            if (d != axis) {
              OPS_CHECK(input0.runtimeshape.dims[d] ===
                        input.runtimeshape.dims[d]);
            }
          }
        }

        // init concatenationParams
        let concatenationParams = new nn_ops.ConcatenationParams;
        concatenationParams.axis = axis;
        concatenationParams.inputs_count = numInputTensors;

        success = nn_ops.concatenationFloat32(concatenationParams, inputShapes, inputValues, 
                                              output.runtimeshape, output.value);
        if (!success) {
          throw new Error('concatenationFloat32 fails');
        }
        inputShapes.delete();
        inputValues.delete();
        concatenationParams.delete();
      } break;
      case OperationCode.FULLY_CONNECTED: {
        allParametersPresent(4, 1);
        let input = operands[inputs[0]];
        let weights = operands[inputs[1]];
        let bias = operands[inputs[2]];
        let activation = operands[inputs[3]].value[0];
        let output = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(weights.runtimeshape.size === 2);

        // init fullyConnectedParams
        let fullyConnectedParams = new nn_ops.FullyConnectedParams;
        fullyConnectedParams.float_activation_range = [float_activation_min, float_activation_max];

        success = nn_ops.fullyConnectedFloat32(fullyConnectedParams, 
                                               input.runtimeshape, input.value, 
                                               weights.runtimeshape, weights.value, 
                                               bias.runtimeshape, bias.value, 
                                               output.runtimeshape, output.value);
        if (!success) {
          throw new Error('fullyConnectedFloat32 fails');
        }
        fullyConnectedParams.delete();
      } break;
      case OperationCode.RESIZE_BILINEAR: {
        allParametersPresent(3, 1);
        let input = operands[inputs[0]];
        let newHeight = operands[inputs[1]].value[0];
        let newWidth = operands[inputs[2]].value[0];  // Dont use newHeight and newWidth since outputShape has been set at first
        let output = operands[outputs[0]];
        let outSizeHeight = output.runtimeshape.dims[1];
        let outSizeWidth = output.runtimeshape.dims[2];

        let outSizeDims = [2];
        let outSizeValue = new Int32Array([outSizeHeight, outSizeWidth]);
        let operand = {
          type: OperandCode.TENSOR_INT32,
          dimensions: outSizeDims,
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: outSizeValue
        }
        let outSizeShape = this._allocateRuntimeShape(operand);
        let outSizeData = this._allocateTensor(operand);

        // Error check
        OPS_CHECK(input.runtimeshape.size <= 4);
        OPS_CHECK(output.runtimeshape.size <= 4);

        // init resizeBilinearParams
        let resizeBilinearParams = new nn_ops.ResizeBilinearParams;
        // default set align_corners to false
        resizeBilinearParams.align_corners = false;
        
        success = nn_ops.resizeBilinearFloat32(resizeBilinearParams, 
                                               input.runtimeshape, input.value, 
                                               outSizeShape, outSizeData,   
                                               output.runtimeshape, output.value);
        if (!success) {
          throw new Error('resizeBilinearFloat32 fails');
        }
        resizeBilinearParams.delete();
        outSizeShape.delete();
        nn_ops._free(outSizeData);
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

  _allocateRuntimeShape(operand) {
    const nn_ops = this._nn_ops;
    let RuntimeShape = new nn_ops.RuntimeShape(operand.dimensions.length);
    RuntimeShape.dims = operand.dimensions;
    return RuntimeShape;
  }

  _deleteAll() {
    this._toDelete.tensorValue.forEach(tensorValue => {
      this._nn_ops._free(tensorValue);
    });
    this._toDelete.tensorShape.forEach(tensorShape => {
      tensorShape.delete();
    });
    this._model._operands = [];
  }
}
