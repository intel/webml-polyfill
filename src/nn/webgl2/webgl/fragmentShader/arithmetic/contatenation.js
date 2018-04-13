/**
 * Create GLSL program for merge.Concatenate layer
 *
 * @param {number} numInputs
 * @param {number[][]} inputShapes
 * @param {number[]} outputShape
 */
export default function contatenation(numInputs, inputShapes, outputShape) {
  const dims = inputShapes.map(shape => shape[1]);
  let offsets = [0];
  dims.forEach((w, i) => {
    offsets.push(dims.slice(0, i + 1).reduce((i, j) => i + j));
  })

  // get offsets
  let getOffset = `
  int n = 0;
  int offset = 0;
  if (out_x >= ${offsets[1]} && out_x < ${offsets[2]}) {
    n = 1;
    offset = ${offsets[1]};
  }`
  if (numInputs > 2) {
    for (let i = 2; i < numInputs; ++i) {
      getOffset += ` else if (out_x >= ${offsets[i]} && out_x < ${offsets[i + 1]}) {
        n = ${i};
        offset = ${offsets[i]};
      }`
    }
  }

  // get out block
  let outBlock = `
  if (n == 0) {
    outColor = vec4(texelFetch(inputs[0], ivec2(out_x, out_y), 0).r);
  }`
  for (let i = 1; i < numInputs; ++i) {
    outBlock += ` else if (n == ${i}) {
      outColor = vec4(texelFetch(inputs[${i}], ivec2(out_x - ${offsets[i]}, out_y), 0).r);
    }`
  }

  const source = `#version 300 es
  precision highp float;
  
  in vec2 outTex;
  uniform sampler2D inputs[${numInputs}];
  out vec4 outColor;
  
  void main() {
    int out_y = int(float(${outputShape[0]}) * outTex.y);
    int out_x = int(float(${outputShape[1]}) * outTex.x);

    ${getOffset}
    ${outBlock}
  }
  `;
  return source;
}

