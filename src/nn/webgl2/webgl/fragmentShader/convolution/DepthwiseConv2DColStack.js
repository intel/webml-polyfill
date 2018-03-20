import { softmax } from "../activation/softmax";

/**
 * Create GLSL program for convolutional.Conv2D layer
 *
 * @param {number[]} outputShape
 * @param {number[]} inputShape
 * @param {number[]} indexMapShape
 * @param {boolean} useBias
 * @param {boolean} [hasFragments]
 */
export default function DepthwiseConv2DDirect(outputShape, inputShape, indexMapShape, useBias, hasFragments) {
    const addBias = useBias ? `sum += texelFetch(bias, ivec2(out_x, 0), 0).r;` : ''
  
    const adjustIndicesForFragments = hasFragments
      ? `int fragmentIndex = int(floor(float(rowIndex) / float(${inputShape[0]})));
        rowIndex = int(mod(float(rowIndex), float(${inputShape[0]})));
        colIndex += fragmentIndex * ${inputShape[1]};`
      : ''
  
    const source = `#version 300 es
  precision highp float;
  precision highp isampler2D;
  
  in vec2 outTex;
  uniform sampler2D x;
  uniform isampler2D indexMap;
  uniform sampler2D kernel;
  uniform sampler2D bias;
  out vec4 outColor;
  
  void main() {
    int out_x = int(float(${outputShape[1]}) * outTex.x);
    int out_y = int(float(${outputShape[0]}) * outTex.y);
    ivec2 kernel_size = textureSize(kernel, 0);
    int length = kernel_size[0];
    // int length = int(float(${indexMapShape[1]}) / float(${outputShape[1]}));
  
    float sum = 0.0;
    for (int i = 0; i < length; ++i) {
      // int index = texelFetch(indexMap, ivec2(i + out_x * length, out_y), 0).r;    
      int index = texelFetch(indexMap, ivec2(out_x, i + out_y * length), 0).r; 
      if (index != -1) {
        int rowIndex = int(floor(float(index) / float(${inputShape[1]})));
        int colIndex = int(mod(float(index), float(${inputShape[1]})));
        // ${adjustIndicesForFragments}
        // sum += texelFetch(x, ivec2(colIndex, rowIndex), 0).r * texelFetch(kernel, ivec2(i + out_x * length, 0), 0).r;
        sum += texelFetch(x, ivec2(colIndex, rowIndex), 0).r * texelFetch(kernel, ivec2(i, out_x), 0).r;
      }
    }
  
    ${addBias}
    outColor = vec4(sum);
    // outColor = vec4(float(out_y));
  }   
  `
    return source
  }
  