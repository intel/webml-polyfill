import * as layers from '../layers'
import Tensor from '../Tensor'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from '../../Enums'
import Layer from '../Layer';

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

/**
 * Get conv2D attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode, kernelCode, biasCode, 
 *                             paddingCodeCode, strideWCode, strideHCode, fuseCode]
 *                             or 
 *                            [inputCode, kernelCode, biasCode, 
 *                             paddingWidthBeginCode, paddingWidthEndCode, paddingHeightBeginCode, paddingHeightEndCode, 
 *                             strideXCode, strideYCode, fuseCode]
 * @param {number[]} outputs - [outputCode].
 */
function GetConv2DAttrs(nnOperands, inputs, outputs) {
  let kernel = nnOperands[inputs[1]];
  let use_bias = false;
  let biasTensor = null;

  if (true) {
    use_bias = true;
    let bias = nnOperands[inputs[2]];
    biasTensor = new Tensor(bias.value, bias.dimensions, OperandCodeMap.get(bias.type));
  }
  let kernelTensor = new Tensor(kernel.value, kernel.dimensions, OperandCodeMap.get(kernel.type));
  let strideHW;
  let padding;
  let activation;
  if (inputs.length === 7) {
    padding = PaddingCodeMap.get(nnOperands[inputs[3]].value[0]);
    strideHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
    activation = FuseCodeMap.get(nnOperands[inputs[6]].value[0]);
  } else if (inputs.length === 10) {
    let pad1 = nnOperands[inputs[3]].value[0];
    let pad2 = nnOperands[inputs[4]].value[0];
    let pad3 = nnOperands[inputs[5]].value[0];
    let pad4 = nnOperands[inputs[6]].value[0];
    padding = [pad3, pad4, pad1, pad2];
    // console.log(padding)
    strideHW = [nnOperands[inputs[7]].value[0], nnOperands[inputs[8]].value[0]];
    activation = FuseCodeMap.get(nnOperands[inputs[9]].value[0]);
  } else {
    throw new Error(`[GetConv2DAttrs] Wrong inputs length: ${inputs.length}`);
  }

  let weights = [kernelTensor, ...(use_bias? [biasTensor] : [])];
  let attrs = {
    inputs: [inputs[0]],
    outputs: outputs,
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
function GetDepthwiseConv2DAttrs (nnOperands, inputs, outputs) {
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
    inputs: [inputs[0]],
    outputs: outputs,
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
 * Get MaxPool2D attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - 
 *                             [inputCode, paddingCodeCode, strideX, strideY, 
 *                             kernelWidthCode, kernelHeightCode, fuseCode].
 *                             or
 *                             [inputCode, paddingWidthBeginCode, paddingWidthEndCode, 
 *                             paddingHeightBeginCode, paddingHeightEnd, strideX, strideY,
 *                             kernelWidthCode, kernelHeightCode, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetMaxPool2DAttrs(nnOperands, inputs, outputs) {
  let padding;
  let strideHW;
  let kernelShapeHW;
  let activation;

  if (inputs.length === 7) {
    padding = PaddingCodeMap.get(nnOperands[inputs[1]].value[0]);
    strideHW = [nnOperands[inputs[2]].value[0], nnOperands[inputs[3]].value[0]];
    kernelShapeHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
    activation = FuseCodeMap.get(nnOperands[inputs[6]].value[0]);
  } else if (inputs.length === 10) {
    let pad1 = nnOperands[inputs[1]].value[0];
    let pad2 = nnOperands[inputs[2]].value[0];
    let pad3 = nnOperands[inputs[3]].value[0];
    let pad4 = nnOperands[inputs[4]].value[0];
    padding = [pad3, pad4, pad1, pad2];
    strideHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[6]].value[0]];
    kernelShapeHW = [nnOperands[inputs[8]].value[0], nnOperands[inputs[7]].value[0]];
    activation = FuseCodeMap.get(nnOperands[inputs[9]].value[0]);
  } else {
    throw new Error(`[GetConv2DAttrs] Wrong inputs length: ${inputs.length}`);
  }

  let attrs = {
    inputs: [inputs[0]],
    outputs: outputs,
    kernel_size: kernelShapeHW,
    strides: strideHW,
    padding: padding,
    activation: activation
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
function GetGlobalAveragePooling2DAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: [inputs[0]],
    outputs: outputs,
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
function GetSoftmaxAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: [inputs[0]],
    outputs: outputs,
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
function GetReshapeAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: [inputs[0]],
    outputs: outputs,
    target_shape: []
  };
  // console.log(attrs);
  return attrs;
}

/**
 * Get Contatenation attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetContatenationAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: inputs.slice(0, inputs.length-1),
    outputs: outputs,
    axis: nnOperands[inputs[inputs.length-1]].value[0]
  };
  // console.log(attrs);
  return attrs;
}

export const OperationCodeToLayersMap = new Map([
  [OperationCode.CONV_2D, layers.Conv2D],
  [OperationCode.DEPTHWISE_CONV_2D, layers.DepthwiseConv2D],
  [OperationCode.MAX_POOL_2D, layers.MaxPool2D],
  [OperationCode.AVERAGE_POOL_2D, layers.GlobalAveragePooling2D],
  [OperationCode.SOFTMAX, layers.Activation],
  [OperationCode.RESHAPE, layers.Reshape],
  [OperationCode.CONCATENATION, layers.Contatenation]
]);

export const OperationCodeAttrsMap = new Map([
  [OperationCode.CONV_2D, GetConv2DAttrs],
  [OperationCode.DEPTHWISE_CONV_2D, GetDepthwiseConv2DAttrs],
  [OperationCode.MAX_POOL_2D, GetMaxPool2DAttrs],
  [OperationCode.AVERAGE_POOL_2D, GetGlobalAveragePooling2DAttrs],
  [OperationCode.SOFTMAX, GetSoftmaxAttrs],
  [OperationCode.RESHAPE, GetReshapeAttrs],
  [OperationCode.CONCATENATION, GetContatenationAttrs]
]);

export const WebGL2SpecialLayers = {
  Input: layers.Input,
  TopClasses: layers.TopClasses
};


// let max = new layers.MaxPool2D({
//   inputs: [0],
//   outputs: [0],
//   kernel_size: [3, 3],
//   strides: [2, 2],
//   padding: 'VALID'
// })

// let input = new Tensor([
//   0,1,2,4,5,
//   0,1,2,4,5,
//   3,4,5,6,3,
//   3,4,5,6,2,
//   6,7,8,9,1,
//   0,1,2,4,5,
//   0,1,2,4,5,
//   3,4,5,6,3,
//   3,4,5,6,2,
//   6,7,8,9,1], [5, 5, 2]);
// input.reshapeTo2D();
// input.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });


// let output = max.call(input)
// output.transferFromGLTexture();
// console.log(output.tensor)