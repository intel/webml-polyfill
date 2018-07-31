export default function matMulDepthwise(fuse) {
  const source = `#version 300 es
  precision highp int;
  precision highp float;
  precision highp sampler2D;
  
  in vec2 outTex;
  uniform sampler2D A;
  uniform sampler2D B;
  uniform sampler2D C;
  uniform int inputChannels;
  uniform int outputChannels;
  uniform int depthMultiplier;
  uniform bool addC;
  out vec4 outColor;
  
  void main() {
    ivec2 A_size = textureSize(A, 0);
    ivec2 B_size = textureSize(B, 0);
    int length = B_size[1];
    int out_x = int(float(outputChannels) * outTex.x);
    int out_y = int(float(A_size[1]) * outTex.y);
    int index = int(floor(float(out_x) / float(depthMultiplier)));
    float sum = 0.0;

    if (index < inputChannels) {
      for (int i = 0; i < length; ++i) {
        float a = texelFetch(A, ivec2(i + index * length, out_y), 0).r;
        float b = texelFetch(B, ivec2(out_x, i), 0).r;
        sum += a * b;
      }
    }
  
    if (addC) {
      sum += texelFetch(C, ivec2(out_x, 0), 0).r;
    }
  
    ${fuse}
    outColor = vec4(sum);
  }`;
  return source;
}
