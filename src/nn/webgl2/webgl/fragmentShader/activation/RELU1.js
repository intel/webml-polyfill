export const RELU1 = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 outTex;
uniform sampler2D x;
out vec4 outColor;

void main() {
  vec4 v = texture(x, vec2(outTex.x, outTex.y));
  outColor = min(max(v, -1.0), 1.0);
}`;
