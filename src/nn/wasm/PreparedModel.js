import getNNOpsInstance from './NNOps'
import { OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime } from '../Enums'
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

    function calculateExplicitPadding(inSize, stride, filterSize, dilationFactor, paddingCode) {
      let paddingHead = 0;
      let paddingTail = 0;

      let dilatedFilterSize = dilationFactor * (filterSize - 1) + 1;

      if (paddingCode === PaddingCode.SAME) {
        let outSize = Math.floor((inSize + stride - 1) / stride);
        let tmp = Math.floor((outSize - 1) * stride + dilatedFilterSize);
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
      } else if (activation === FuseCode.NONE) {
        activation_min = nn_ops.LOWEST;
        activation_max = nn_ops.MAX;
      } else {
        throw new Error("Unsupported fused activation function.");
      }
      return { activation_min, activation_max };
    }

    function sameShape(input1, input2) {
      if (input1.type != input2.type || 
        input1.runtimeshape.DimensionsCount() != input2.runtimeshape.DimensionsCount()) {
        return false;
      }
      for (let i = 0; i < input1.runtimeshape.DimensionsCount(); i++) {
        if (input1.runtimeshape.Dims(i) != input2.runtimeshape.Dims(i)) {
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
        OPS_CHECK(in1.runtimeshape.DimensionsCount() <= 4 && in2.runtimeshape.DimensionsCount() <= 4);

        // init arithmeticParams
        let arithmeticParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max
        }
        
        let needBroadCast = !sameShape(in1, in2);
        if (needBroadCast) {
          nn_ops.broadCastAddFloat32(arithmeticParams,
                                     in1.runtimeshape, in1.value,
                                     in2.runtimeshape, in2.value,
                                     out.runtimeshape, out.value);
        } else {
          nn_ops.addFloat32(arithmeticParams,
                            in1.runtimeshape, in1.value,
                            in2.runtimeshape, in2.value,
                            out.runtimeshape, out.value);
        }
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
        OPS_CHECK(in1.runtimeshape.DimensionsCount() <= 4 && in2.runtimeshape.DimensionsCount() <= 4);

        // init arithmeticParams
        let arithmeticParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max
        }

        let needBroadCast = !sameShape(in1, in2);
        if (needBroadCast) {
          nn_ops.broadCastMulFloat32(arithmeticParams,
                                     in1.runtimeshape, in1.value,
                                     in2.runtimeshape, in2.value,
                                     out.runtimeshape, out.value);
        } else {
          nn_ops.mulFloat32(arithmeticParams,
                            in1.runtimeshape, in1.value,
                            in2.runtimeshape, in2.value,
                            out.runtimeshape, out.value);
        }
      } break;
      case OperationCode.CONV_2D:
      case OperationCode.ATROUS_CONV_2D: {
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
        let dilationWidth, dilationHeight;
        let filterWidth = filter.runtimeshape.Dims(2);
        let filterHeight = filter.runtimeshape.Dims(1);
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          if (op === OperationCode.CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          [paddingLeft, paddingRight] = 
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, dilationWidth, paddingCode);
          [paddingTop, paddingBottom] = 
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, dilationHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        // init im2col operand
        let outBatch = output.runtimeshape.Dims(0);
        let outHeight = output.runtimeshape.Dims(1);
        let outWidth = output.runtimeshape.Dims(2);
        let inDepth = input.runtimeshape.Dims(3);
        let im2colDepth = filterWidth * filterHeight * inDepth;
        let im2colDims = [outBatch, outHeight, outWidth, im2colDepth];
        let im2colValue = new Float32Array(product(im2colDims));
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

        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(filter.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(bias.runtimeshape.DimensionsCount() === 1);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        OPS_CHECK(filter.runtimeshape.Dims(0) === bias.runtimeshape.Dims(0));
        OPS_CHECK(filter.runtimeshape.Dims(3) === input.runtimeshape.Dims(3));

        // init convParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let convParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          dilation_width_factor: dilationWidth,
          dilation_height_factor: dilationHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max
        }

        nn_ops.convFloat32(convParams, 
                           input.runtimeshape, input.value, 
                           filter.runtimeshape, filter.value, 
                           bias.runtimeshape, bias.value, 
                           output.runtimeshape, output.value,
                           im2colShape, im2colData);
        im2colShape.delete();
        nn_ops._free(im2colData);
      } break;
      case OperationCode.DEPTHWISE_CONV_2D:
      case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
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
        let dilationWidth, dilationHeight;
        let depthMultipler;
        let activation;
        if (inCount === 11) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          let filterWidth = filter.runtimeshape.Dims(2);
          let filterHeight = filter.runtimeshape.Dims(1);

          [paddingLeft, paddingRight] = 
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, dilationWidth, paddingCode);
          [paddingTop, paddingBottom] = 
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, dilationHeight, paddingCode);
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

        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(filter.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(bias.runtimeshape.DimensionsCount() === 1);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        OPS_CHECK(filter.runtimeshape.Dims(3) === bias.runtimeshape.Dims(0));

        // init depthwiseParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let depthwiseParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          dilation_width_factor: dilationWidth,
          dilation_height_factor: dilationHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          depth_multiplier: depthMultipler
        }
        nn_ops.depthwiseConvFloat32(depthwiseParams, 
                                    input.runtimeshape, input.value, 
                                    filter.runtimeshape, filter.value, 
                                    bias.runtimeshape, bias.value, 
                                    output.runtimeshape, output.value);
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

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          [paddingLeft, paddingRight] = 
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, 1, paddingCode);
          [paddingTop, paddingBottom] = 
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, 1, paddingCode);
        }
        let output = operands[outputs[0]];

        let float_activation_min = calculateActivationRangeFloat(activation).activation_min;
        let float_activation_max = calculateActivationRangeFloat(activation).activation_max;

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        // init poolParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let poolParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          filter_width: filterWidth,
          filter_height: filterHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max
        }

        if (op === OperationCode.AVERAGE_POOL_2D) {
          nn_ops.averagePoolFloat32(poolParams, 
                                    input.runtimeshape, input.value,
                                    output.runtimeshape, output.value);
        } else if (op === OperationCode.MAX_POOL_2D) {
          nn_ops.maxPoolFloat32(poolParams, 
                                input.runtimeshape, input.value,
                                output.runtimeshape, output.value);
        }
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
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);

        // init softmaxParams
        let softmaxParams = {
          beta: beta
        }

        nn_ops.softmaxFloat32(softmaxParams, 
                              input.runtimeshape, input.value, 
                              output.runtimeshape, output.value);
      } break;
      case OperationCode.RESHAPE: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let targetShape = operands[inputs[1]];  // Dont use targetShape since 
                                                // outputShape has been set at first
        let output = operands[outputs[0]];

        let inputDims = [];
        let  outputDims = [];
        for (let i = 0; i < input.runtimeshape.DimensionsCount(); ++i) {
          inputDims.push(input.runtimeshape.Dims(i));
        }
        for (let i = 0; i < output.runtimeshape.DimensionsCount(); ++i) {
          outputDims.push(output.runtimeshape.Dims(i));
        }

        // Error check
        let numInputElements = product(inputDims);
        let numOutputElements = product(outputDims);
        OPS_CHECK(numInputElements === numOutputElements);

        nn_ops.reshapeFloat32(input.runtimeshape, input.value,  
                              output.runtimeshape, output.value);
      } break;
      case OperationCode.CONCATENATION: {
        if (outputs.length < 1 || inputs.length < 2) {
          throw new Error('Invalid inputs or outputs');
        }
        let numInputTensors = inputs.length - 1;
        let axis = operands[inputs[numInputTensors]].value[0];
        let input0 = operands[inputs[0]];
        let num_dimensions = input0.runtimeshape.DimensionsCount();
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
        for (let i = 1; i < numInputTensors; ++i) {
          let input = operands[inputs[i]];
          OPS_CHECK(input.runtimeshape.DimensionsCount() === num_dimensions);
          OPS_CHECK(input.type === input_type);
          for (let d = 0; d < num_dimensions; ++d) {
            if (d != axis) {
              OPS_CHECK(input0.runtimeshape.Dims(d) ===
                        input.runtimeshape.Dims(d));
            }
          }
        }

        // init concatenationParams
        let concatenationParams = {
          axis: axis,
          inputs_count: numInputTensors
        }

        nn_ops.concatenationFloat32(concatenationParams, inputShapes, inputValues, 
                                    output.runtimeshape, output.value);
        inputShapes.delete();
        inputValues.delete();
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
        OPS_CHECK(weights.runtimeshape.DimensionsCount() === 2);

        // init fullyConnectedParams
        let fullyConnectedParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max
        }

        nn_ops.fullyConnectedFloat32(fullyConnectedParams, 
                                     input.runtimeshape, input.value, 
                                     weights.runtimeshape, weights.value, 
                                     bias.runtimeshape, bias.value, 
                                     output.runtimeshape, output.value);
      } break;
      case OperationCode.RESIZE_BILINEAR: {
        let inCount = inputs.length;
        if (inCount !== 3 && inCount !== 4) {
          throw new Error(`Invalid parameters number of resize bilinear ${op}`);
        }
        allParametersPresent(inCount, 1);
        let input = operands[inputs[0]];
        let newHeight = operands[inputs[1]].value[0]; // Dont use newHeight and newWidth
        let newWidth = operands[inputs[2]].value[0];  // since outputShape has been set at first
        // init resizeBilinearParams
        // default set align_corners to false
        let resizeBilinearParams = {
          align_corners: false
        };
        if (inCount === 4) {
          resizeBilinearParams.align_corners =
              operands[inputs[3]].value[0] !== 0;
        }
        let output = operands[outputs[0]];
        let outSizeHeight = output.runtimeshape.Dims(1);
        let outSizeWidth = output.runtimeshape.Dims(2);

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
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() <= 4);
        
        nn_ops.resizeBilinearFloat32(resizeBilinearParams, 
                                     input.runtimeshape, input.value, 
                                     outSizeShape, outSizeData,   
                                     output.runtimeshape, output.value);
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
      nn_ops.HEAPF32.set(data, ptr >> 2);
    } else if (type === OperandCode.TENSOR_INT32) {
      nn_ops.HEAP32.set(data, ptr >> 2);
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
    for (let i = 0; i < RuntimeShape.DimensionsCount(); ++i) {
      RuntimeShape.SetDim(i, operand.dimensions[i]);
    }
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
