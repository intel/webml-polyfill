/**
 * Create GLSL program for TopClasses layer
 *
 * @param {number} reduceNums
 * @param {number} reduceLen
 * @param {number} index
 * @param {number} x_width
 */

export default function reduceClasses(reduceNums, reduceLen, index, x_width) {
  let high_index;
  if (index == 0) {
    high_index = `i`;
  } else {
    high_index = `int(texelFetch(x, ivec2(i, 1), 0).r)`;
  }

  const source = `#version 300 es
  precision highp float;
  precision highp sampler2D;

  in vec2 outTex;
  uniform sampler2D x;
  out vec4 outColor;

  void main() {
    int out_x = int(float(${reduceLen}) * outTex.x);
    int out_y = int(2.0 * outTex.y);
    float high;
    int high_index;
    // outColor = vec4(float(out_x));
    if (out_y == 0) {
      for (int i = out_x * ${reduceNums}; i < min((out_x + 1) * ${reduceNums}, ${x_width}); ++i) {
        float x_pixelR = texelFetch(x, ivec2(i, 0), 0).r;
        if (i == out_x * ${reduceNums} || x_pixelR > high) {
          high = x_pixelR;
        }
      }
      outColor = vec4(high);
    } else {
      for (int i = out_x * ${reduceNums}; i < min((out_x + 1) * ${reduceNums}, ${x_width}); ++i) {
        float x_pixelR = texelFetch(x, ivec2(i, 0), 0).r;
        if (i == out_x * ${reduceNums} || x_pixelR > high) {
          high = x_pixelR;
          high_index = ${high_index};
        }
      }
      outColor = vec4(float(high_index));
    }
  }`;
  return source;
}
  