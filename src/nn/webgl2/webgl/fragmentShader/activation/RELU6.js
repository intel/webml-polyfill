export const RELU6 = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 outTex;
uniform sampler2D x;
out vec4 outColor;

void main() {
  vec4 v = texture(x, vec2(outTex.x, outTex.y));
  outColor = min(max(v, 0.0), 6.0);
}`;
