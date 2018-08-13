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
 * Get add attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [input0Code, input1Code, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetAddAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: inputs.slice(0, 2),
    outputs: outputs,
    activation: FuseCodeMap.get(nnOperands[inputs[2]].value[0])
  };
  return attrs;
}

/**
 * Get mul attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [input0Code, input1Code, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetMulAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: inputs.slice(0, 2),
    outputs: outputs,
    activation: FuseCodeMap.get(nnOperands[inputs[2]].value[0])
  };
  return attrs;
}

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
  let strideHW;
  let padding;
  let activation;
  let depthMultiplier;
  if (inputs.length === 8) {
    padding = PaddingCodeMap.get(nnOperands[inputs[3]].value[0]);
    strideHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
    depthMultiplier = nnOperands[inputs[6]].value[0];
    activation = FuseCodeMap.get(nnOperands[inputs[7]].value[0]);
  } else if (inputs.length === 11) {
    let pad1 = nnOperands[inputs[3]].value[0];
    let pad2 = nnOperands[inputs[4]].value[0];
    let pad3 = nnOperands[inputs[5]].value[0];
    let pad4 = nnOperands[inputs[6]].value[0];
    padding = [pad3, pad4, pad1, pad2];
    // console.log(padding)
    strideHW = [nnOperands[inputs[8]].value[0], nnOperands[inputs[7]].value[0]];
    depthMultiplier = nnOperands[inputs[9]].value[0];
    activation = FuseCodeMap.get(nnOperands[inputs[10]].value[0]);
  } else {
    throw new Error(`[GetDepthwiseConv2DAttrs] Wrong inputs length: ${inputs.length}`);
  }

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
 *                             [inputCode, paddingCodeCode, strideWCode, strideHCode, 
 *                             kernelWidthCode, kernelHeightCode, fuseCode].
 *                             or
 *                             [inputCode, paddingWidthBeginCode, paddingWidthEndCode, 
 *                             paddingHeightBeginCode, paddingHeightEnd, strideWCode, strideHCode,
 *                             kernelWidthCode, kernelHeightCode, fuseCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetPool2DAttrs(nnOperands, inputs, outputs) {
  let padding;
  let strideHW;
  let kernelShapeHW;
  let activation;

  if (inputs.length === 7) {
    padding = PaddingCodeMap.get(nnOperands[inputs[1]].value[0]);
    strideHW = [nnOperands[inputs[3]].value[0], nnOperands[inputs[2]].value[0]];
    kernelShapeHW = [nnOperands[inputs[5]].value[0], nnOperands[inputs[4]].value[0]];
    activation = FuseCodeMap.get(nnOperands[inputs[6]].value[0]);
  } else if (inputs.length === 10) {
    let pad1 = nnOperands[inputs[1]].value[0];
    let pad2 = nnOperands[inputs[2]].value[0];
    let pad3 = nnOperands[inputs[3]].value[0];
    let pad4 = nnOperands[inputs[4]].value[0];
    padding = [pad3, pad4, pad1, pad2];
    strideHW = [nnOperands[inputs[6]].value[0], nnOperands[inputs[5]].value[0]];
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
    beta: nnOperands[inputs[1]].value[0],
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
 * Get Concatenation attributes.
 * 
 * @param {Object[]} nnOperands - An array of operands.
 * @param {number[]} inputs - [inputCode].
 * @param {number[]} outputs - [outputCode].
 */
function GetConcatenationAttrs(nnOperands, inputs, outputs) {
  let attrs = {
    inputs: inputs.slice(0, inputs.length-1),
    outputs: outputs,
    axis: nnOperands[inputs[inputs.length-1]].value[0]
  };
  // console.log(attrs);
  return attrs;
}

export const OperationCodeToLayersMap = new Map([
  [OperationCode.ADD, layers.Add],
  [OperationCode.MUL, layers.Mul],
  [OperationCode.CONV_2D, layers.Conv2D],
  [OperationCode.DEPTHWISE_CONV_2D, layers.DepthwiseConv2D],
  [OperationCode.MAX_POOL_2D, layers.MaxPool2D],
  [OperationCode.AVERAGE_POOL_2D, layers.AveragePool2D],
  [OperationCode.SOFTMAX, layers.Activation],
  [OperationCode.RESHAPE, layers.Reshape],
  [OperationCode.CONCATENATION, layers.Concatenation]
]);

export const OperationCodeAttrsMap = new Map([
  [OperationCode.ADD, GetAddAttrs],
  [OperationCode.MUL, GetMulAttrs],
  [OperationCode.CONV_2D, GetConv2DAttrs],
  [OperationCode.DEPTHWISE_CONV_2D, GetDepthwiseConv2DAttrs],
  [OperationCode.MAX_POOL_2D, GetPool2DAttrs],
  [OperationCode.AVERAGE_POOL_2D, GetPool2DAttrs],
  [OperationCode.SOFTMAX, GetSoftmaxAttrs],
  [OperationCode.RESHAPE, GetReshapeAttrs],
  [OperationCode.CONCATENATION, GetConcatenationAttrs]
]);

export const WebGL2SpecialLayers = {
  Input: layers.Input,
  TopClasses: layers.TopClasses,
  FeatureMapConcate: layers.FeatureMapConcate
};

// webgl2 operation test

// let inputShape = [100, 50, 3]
// let inputShapeLen = inputShape.reduce((i, j) => i * j);
// let inputValue = [];
// for (let i =  0; i < inputShapeLen; ++i) {
//   inputValue.push(1/10000);
// }
// let inputTensor = new Tensor(inputValue, inputShape, Float32Array);
// inputTensor.reshapeTo2D();
// inputTensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });

// let max = new layers.MaxPool2D({
//   inputs: [0],
//   outputs: [0],
//   kernel_size: [3, 3],
//   strides: [1, 1],
//   padding: [1,1,1,1]
// })

// let kernelValue = [];
// let kernelShape = [1, 3, 3, inputShape[2]];
// let kernelValueLen = kernelShape.reduce((i, j) => i * j);
// for (let i =  0; i < kernelValueLen; ++i) {
//   kernelValue.push(i);
// }

// let biasValue = [];
// let biasValueLen = kernelShape[0];
// for (let i =  0; i < biasValueLen; ++i) {
//   biasValue.push(i);
// }

// let use_bias = false;
// let biasTensor = new Tensor(biasValue, [biasValueLen], Float32Array);
// let kernelTensor = new Tensor(kernelValue, kernelShape, Float32Array);
// let weights = [kernelTensor, ...(use_bias? [biasTensor] : [])];

// let conv = new layers.Conv2D({
//   inputs: [0],
//   outputs: [0],
//   filters: kernelShape[0],
//   kernel_size: kernelShape.slice(1,3),
//   strides: [1, 1],
//   padding: 'VALID',
//   activation: 'RELU',
//   use_bias: use_bias,
//   weights: weights
// })

// console.log(inputTensor)
// let maxOutput = max.call(inputTensor);
// maxOutput.transferFromGLTexture();
// console.log('max', maxOutput.tensor)

// let conOutput = conv.call(inputTensor);
// conOutput.transferFromGLTexture();
// console.log('conv', conOutput)

// conOutput = conv.call(conOutput);
// conOutput.transferFromGLTexture();
// console.log('conv', conOutput)

// let input1Shape = [100, 100, 2]
// let input1ShapeLen = input1Shape.reduce((i, j) => i * j);
// let input1Value = [];
// for (let i =  0; i < input1ShapeLen; ++i) {
//   input1Value.push(i/100);
// }
// let input1Tensor = new Tensor(input1Value, input1Shape, Float32Array);
// input1Tensor.reshapeTo2D();
// input1Tensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });

// let input2Shape = [100, 100, 2]
// let input2ShapeLen = input2Shape.reduce((i, j) => i * j);
// let input2Value = [];
// for (let i =  0; i < input2ShapeLen; ++i) {
//   input2Value.push(i/100);
// }
// let input2Tensor = new Tensor(input2Value, input2Shape, Float32Array);
// input2Tensor.reshapeTo2D();
// input2Tensor.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });

// let concate = new layers.Concatenation({
//   inputs: [0],
//   outputs: [0]
// })

// let output1 = concate.call([input1Tensor, input2Tensor]);
// output1.transferFromGLTexture();
// console.log('concate', output1)

// let inputLayer1 = new layers.Input()
// let inputLayer2 = new layers.Input()
// let out1 = inputLayer1.call(input1Value, input1Shape, Float32Array)
// let out2 = inputLayer2.call(input2Value, input2Shape, Float32Array)
// let concatetest = new layers.Concatenation({
//   inputs: [0],
//   outputs: [0]
// })

// let output2 = concate.call([out1, out2]);
// output2.transferFromGLTexture();
// console.log('inputLayer concate', output2)

// let output3 = new Tensor(output2.tensor.data, [100, 100, 4], Float32Array);
// // output2.is2DReshaped = false;
// let conOutput = conv.call(output2);
// conOutput.transferFromGLTexture();
// console.log('conv', conOutput)