import * as layers from '../layers'
import Tensor from '../Tensor'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from '../../Enums'

const OperandCodeMap = new Map([
  [OperandCode.TENSOR_FLOAT32, Float32Array],
  [OperandCode.TENSOR_INT32, Int32Array]
]);

const PaddingCodeMap = new Map([
  [PaddingCode.SAME, 'SAME'],
  [PaddingCode.VALID, 'VALID']
]);

const FuseCodeMap = new Map([
  [FuseCode.NONE, 'NONE'],
  [FuseCode.RELU, 'RELU'],
  [FuseCode.RELU1, 'RELU1'],
  [FuseCode.RELU6, 'RELU6'],
]);

// // tensors in and tensors out 
// function GetInputsOutputs(nnOperands, inputs, output) {
//   let inputTensors = [];
//   // inputs.forEach(input => {
//   //   let inputOp = nnOperands[input];
//   //   inputTensors.push(new Tensor(inputOp.value, inputOp.dimensions, OperandCodeMap.get(inputOp.type)));
//   // });
//   let inputOp = nnOperands[inputs[0]];
//   // console.log([], inputOp.dimensions, OperandCodeMap.get(inputOp.type));
//   inputTensors.push(new Tensor([], inputOp.dimensions, OperandCodeMap.get(inputOp.type)));

//   let outputTensors = [];
//   // outputs.forEach(output => {
//     // let outputOp = nnOperands[output];
//     // outputTensors.push(new Tensor([], outputOp.dimensions, OperandCodeMap.get(outputOp.type)));
//   // });
//   let outputOp = nnOperands[outputs[0]];
//   outputTensors.push(new Tensor([], outputOp.dimensions, OperandCodeMap.get(outputOp.type)))
//   console.log([], outputOp.dimensions, OperandCodeMap.get(outputOp.type));
//   return [inputTensors, outputTensors];
// }


/**
 * Get conv2D attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode, kernelCode, biasCode, 
 *                             paddingCodeCode, strideWCode, strideHCode, fuseCode]
 * @param {number[]} outputs - [outputCode].
 */
function GetConv2DAttrs(nnOperands, inputs, output) {
  let kernel = nnOperands[inputs[1]];
  let use_bias = false;
  let biasTensor = null;

  if (true) {
    use_bias = true;
    let bias = nnOperands[inputs[2]];
    biasTensor = new Tensor(bias.value, bias.dimensions, OperandCodeMap.get(bias.type));
  }
  let kernelTensor = new Tensor(kernel.value, kernel.dimensions, OperandCodeMap.get(kernel.type));
  let strideHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
  let padding = PaddingCodeMap.get(nnOperands[inputs[3]].value[0]);
  let activation = FuseCodeMap.get(nnOperands[inputs[6]].value[0]);
  let weights = [kernelTensor, ...(use_bias? [biasTensor] : [])];
  let attrs = {
    filters: kernel.dimensions[0],
    kernel_size: kernel.dimensions.slice(1,3),
    strides: strideHW,
    padding: padding,
    activation: activation,
    use_bias: use_bias,
    weights: weights
  };
  // console.log(attrs);
  return attrs;
}

/**
 * Get depthwiseConv2D attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode, kernelCode, biasCode, paddingCodeCode,  
 *                             strideWCode, strideHCode, depthMultiplierCode, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetDepthwiseConv2DAttrs (nnOperands, inputs, output) {
  let kernel = nnOperands[inputs[1]];
  let use_bias = false;
  let biasTensor = null;

  if (true) {
    use_bias = true;
    let bias = nnOperands[inputs[2]];
    biasTensor = new Tensor(bias.value, bias.dimensions, OperandCodeMap.get(bias.type));
  }

  let kernelTensor = new Tensor(kernel.value, kernel.dimensions, OperandCodeMap.get(kernel.type));
  let strideHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
  let padding = PaddingCodeMap.get(nnOperands[inputs[3]].value[0]);
  let depthMultiplier = nnOperands[inputs[6]].value[0];
  let activation = FuseCodeMap.get(nnOperands[inputs[7]].value[0]);
  let weights = [kernelTensor, ...(use_bias? [biasTensor] : [])];
  let attrs = {
    filters: kernel.dimensions[0],
    kernel_size: kernel.dimensions.slice(1,3),
    strides: strideHW,
    padding: padding,
    depthMultiplier : depthMultiplier,
    activation: activation,
    use_bias: use_bias,
    weights: weights
  };
  // console.log(attrs);
  return attrs;
}

/**
 * Get average pooling attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode, paddingCodeCode, strideWCode, strideHCode, 
 *                             filterWidthCode, filterHeightCode, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetGlobalAveragePooling2DAttrs(nnOperands, inputs, output) {
  let attrs = {
    data_format: 'HWC'
  };
  // console.log(attrs);
  return attrs;
}

/**
 * Get softmax attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode, betaCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetSoftmaxAttrs(nnOperands, inputs, output) {
  let attrs = {
    activation: 'softmax'
  };
  // console.log(attrs);
  return attrs;
}

/**
 * Get reshape attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetReshapeAttrs(nnOperands, inputs, output) {
  let attrs = {
    target_shape: []
  };
  // console.log(attrs);
  return attrs;
}

export const OperationCodeToLayersMap = new Map([
  [OperationCode.CONV_2D, layers.Conv2D],
  [OperationCode.DEPTHWISE_CONV_2D, layers.DepthwiseConv2D],
  [OperationCode.AVERAGE_POOL_2D, layers.GlobalAveragePooling2D],
  [OperationCode.SOFTMAX, layers.Activation],
  [OperationCode.RESHAPE, layers.Reshape]
]);

export const OperationCodeAttrsMap = new Map([
  [OperationCode.CONV_2D, GetConv2DAttrs],
  [OperationCode.DEPTHWISE_CONV_2D, GetDepthwiseConv2DAttrs],
  [OperationCode.AVERAGE_POOL_2D, GetGlobalAveragePooling2DAttrs],
  [OperationCode.SOFTMAX, GetSoftmaxAttrs],
  [OperationCode.RESHAPE, GetReshapeAttrs]
]);

export const WebGL2SpecialLayers = {
  Input: layers.Input,
  TopClasses: layers.TopClasses
};