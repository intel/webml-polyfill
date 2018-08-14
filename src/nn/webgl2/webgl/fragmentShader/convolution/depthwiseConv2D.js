/**
 * Create GLSL program for DepthwiseConv2 layer
 *
 * @param {number} inputChannels
 * @param {number} outputChannels
 * @param {number} depthMultiplier
 * @param {boolean} useBias
 * @param {boolean} [hasFragments]
 */
export default function depthwiseConv2D(inputChannels, outputChannels, depthMultiplier, useBias, hasSlices, fuse) {
  const addBias = useBias ? `sum += texelFetch(bias, ivec2(out_x, 0), 0).r;` : '';

  const adjustIndicesForSlices = hasSlices
  ? `ivec2 inputSize = textureSize(x, 0);
      int sliceIndex = int(floor(float(index) / float(inputSize[1])));
      index = int(mod(float(index), float(inputSize[1])));
      int fetch_x = sliceIndex * ${inputChannels} + in_x;`
  : `int fetch_x = in_x;`;

  const source = `#version 300 es
  precision highp int;
  precision highp float;
  precision highp isampler2D;
  precision highp sampler2D;
  
  in vec2 outTex;
  uniform sampler2D x;
  uniform isampler2D indexMap;
  uniform sampler2D kernel;
  uniform sampler2D bias;
  out vec4 outColor;
  
  void main() {
    ivec2 indexMapSize = textureSize(indexMap, 0);
    int out_x = int(float(${outputChannels}) * outTex.x);
    int out_y = int(float(indexMapSize[1]) * outTex.y);
    ivec2 kernelSize = textureSize(kernel, 0);
    int convSize = kernelSize[1];
    int in_x = int(floor(float(out_x) / float(${depthMultiplier})));
    float sum = 0.0;

    if(in_x < ${inputChannels}) {
      for (int i = 0; i < convSize; ++i) {
        int index = texelFetch(indexMap, ivec2(i, out_y), 0).r; 
        if (index != -1) {
          ${adjustIndicesForSlices}
          sum += texelFetch(x, ivec2(fetch_x, index), 0).r * texelFetch(kernel, ivec2(out_x, i), 0).r;
        }
      }
    }
    ${addBias}
    ${fuse}
    outColor = vec4(sum);
  }`;
    return source;
  }
  