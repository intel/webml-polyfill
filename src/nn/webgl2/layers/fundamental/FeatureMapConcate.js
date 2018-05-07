import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as tensorUtils from '../../utils/tensorUtils'
import webgl2 from '../../WebGL2'
import ops from 'ndarray-ops'

 /**
 * FeatureMapConcate merge layer class, extends abstract _Merge class
 */
export default class FeatureMapConcate extends Layer {
  /**
   * Creates a FeatureMapConcate merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'FeatureMapConcate';
    this.inputsShapeList = [];
    this.outH = [];
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  call(inputs, outputs) {
    let boxSize = inputs[inputs.length - 2].textureShape[1];
    let numClasses = inputs[inputs.length - 1].textureShape[1];
    
    if (this.outH.length === 0) {
      for (let i = 0; i < inputs.length; ++i) {
        if (i % 2 === 0) {
          this.outH.push(inputs[i].textureShape[0]);
        }
        this.inputsShapeList.push(inputs[i].textureShape)
      }
      if (this.inputsShapeList.length !== this.outH.length * 2) {
        this.throwError('Wrong input length.');
      }
    }

    if (!this.output) {
      const outputTextureShape = [this.outH.reduce((i, j) => i + j), boxSize + numClasses];
      this.output = new Tensor([], outputTextureShape);
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
    }
    const gl = webgl2.context;
    const textureOptions = webgl2.getTextureOptions(inputs[0].textureType, inputs[0].textureFormat);
    const { textureTarget, textureInternalFormat, textureFormat, textureType } = textureOptions;

    if (!webgl2.featureMapFramebuffer) {
      webgl2.featureMapFramebuffer = gl.createFramebuffer();
      webgl2.toDelete.framebuffers.push(webgl2.featureMapFramebuffer);
    }

    gl.bindTexture(textureTarget, this.output.texture);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, webgl2.featureMapFramebuffer);
    
    let Yoffset = 0;
    let input;
    for (let i = 0; i < this.outH.length; ++i) {
      input = inputs[i * 2];
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, 
                              gl.TEXTURE_2D, input.texture, 0);
      gl.copyTexSubImage2D(
        textureTarget,
        0,
        0,
        Yoffset,
        0,
        0,
        input.textureShape[1],
        input.textureShape[0]
      )
      input = inputs[i * 2 + 1];
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, 
                              gl.TEXTURE_2D, input.texture, 0);
      gl.copyTexSubImage2D(
        textureTarget,
        0,
        boxSize,
        Yoffset,
        0,
        0,
        input.textureShape[1],
        input.textureShape[0]
      )
      Yoffset += this.outH[i];
    }
    this.output.transferFromGLTexture();

    let Hoffset = 0;
    let outputBoxBuffer;
    let boxIndex;
    let outputClassBuffer;
    let classIndex;
    let H;
    for (let i = 0; i < this.inputsShapeList.length / 2; ++i) {
      boxIndex = i * 2;
      outputBoxBuffer = new Tensor(outputs.get(boxIndex).buffer, this.inputsShapeList[boxIndex]);
      classIndex = boxIndex + 1;
      outputClassBuffer = new Tensor(outputs.get(boxIndex + 1).buffer, this.inputsShapeList[boxIndex + 1]);
      H = this.inputsShapeList[boxIndex][0];
      ops.assign(
        outputBoxBuffer.tensor,
        this.output.tensor
          .hi(Hoffset + H, this.inputsShapeList[boxIndex][1])
          .lo(Hoffset, 0)
      );
      ops.assign(
        outputClassBuffer.tensor,
        this.output.tensor
          .hi(Hoffset + H, this.inputsShapeList[boxIndex][1] + this.inputsShapeList[classIndex][1])
          .lo(Hoffset, this.inputsShapeList[boxIndex][1])
      );
      Hoffset += H;
    }
  }
}
