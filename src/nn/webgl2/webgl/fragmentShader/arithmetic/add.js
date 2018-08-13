export default function add(fuse) {
  const source = `#version 300 es
  precision highp float;
  precision highp sampler2D;
  
  in vec2 outTex;
  uniform sampler2D A;
  uniform sampler2D B;
  out vec4 outColor;
  
  void main() {
    ivec2 A_size = textureSize(A, 0);
    int out_x = int(float(A_size[0]) * outTex.x);
    int out_y = int(float(A_size[1]) * outTex.y);
  
    float a = texelFetch(A, ivec2(out_x, out_y), 0).r;
    float b = texelFetch(B, ivec2(out_x, out_y), 0).r;
    float sum = a + b;
    ${fuse}
    outColor = vec4(sum);
  }`;
  return source;
}
