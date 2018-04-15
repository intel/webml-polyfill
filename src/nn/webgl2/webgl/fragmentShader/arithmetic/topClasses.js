/**
 * Create GLSL program for TopClasses layer
 *
 * @param {number} numTopC
 * @param {number} length
 */

export default function topClasses(numTopC, length) {
  const source = ` #version 300 es
  precision highp float;
  precision highp sampler2D;

  in vec2 outTex;
  uniform sampler2D x;
  out vec4 outColor;

  void main() {
    ivec2 size = textureSize(x, 0);
    int out_x = int(float(size[0]) * outTex.x);
    if (out_x < 2 * ${numTopC}) {
      int numTop = out_x;
      if (numTop >= ${numTopC}) {
        numTop = numTop - ${numTopC};
      }

      float textureCopy[${length}];
      int index[${length}];
      for (int i = 0; i < size[0]; ++i) {
        textureCopy[i] = texelFetch(x, ivec2(i, 0), 0).r;
        index[i] = i;
      }
      int high = size[0] - 1;
      int low = 0;
      while (low < high) {
        int i = low;
        int j = high;
        float pixel = textureCopy[low];
        int pixelIndex = index[low];
        while (i < j) {
          while (i < j && textureCopy[j] < pixel)
            --j;
          if (i < j) {
            textureCopy[i] = textureCopy[j]; 
            index[i] = index[j];
            i++;
          }
          while (i < j && textureCopy[i] > pixel)
            ++i;
          if (i < j) {
            textureCopy[j] = textureCopy[i]; 
            index[j] = index[i];
            j--;
          }
        }
        textureCopy[i] = pixel;
        index[i] = pixelIndex;
        if (i == numTop) {
          low = high;
        }
        else if (i < numTop)
          low = i + 1;
        else
          high = i - 1;
      }
      if (out_x < ${numTopC}) {
        outColor = vec4(textureCopy[numTop]);
      } else {
        outColor = vec4(float(index[numTop]));
      }
    } else {
      outColor = vec4(0.0);
    }
  }`;
  return source;
}
  