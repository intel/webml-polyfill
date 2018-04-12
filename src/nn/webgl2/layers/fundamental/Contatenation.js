import Layer from '../../Layer'
import Tensor from '../../Tensor'
import * as tensorUtils from '../../utils/tensorUtils'
import contatenation from '../../webgl/fragmentShader/arithmetic/contatenation'
import webgl2 from '../../WebGL2'

 /**
 * Contatenation merge layer class, extends abstract _Merge class
 */
export default class Contatenation extends Layer {
  /**
   * Creates a Contatenation merge layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs);
    this.name = 'Contatenation';
    const { axis = -1 } = attrs;
    this.axis = axis;
  }

  /**
   * GPU call
   *
   * @param {Tensor[]} inputs
   */
  call(inputs) {
    // C axis is 3 in NHWC layout
    // no mini-batch axis here, so we subtract 1 if given axis > 0
    this.concatAxis = this.axis <= 0 ? this.axis + inputs.length: this.axis - 1;

    inputs.forEach(input => {
      if (!input.texture && !input.textureSlices) {
        input.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      }
    })

    const outputTextureShape = inputs[0].textureShape.slice();
    // _concatAxis = 1 for 2D Texture
    let _concatAxis = 1;
    // create output textures if doesn't already exist
    outputTextureShape[_concatAxis] = inputs.map(input => input.textureShape[_concatAxis])
                                            .reduce((i, j) => i + j);
    if (!this.output) {
      this.output = new Tensor([], outputTextureShape);
      this.output.createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
      if (inputs[0].is1D) {
        this.output.is1D = inputs[0].is1D;
      } else if (inputs[0].is2DReshaped) {
        this.output.is2DReshaped = inputs[0].is2DReshaped;
        this.output.originalShape = inputs[0].originalShape.slice();
        this.output.originalShape[this.concatAxis] = inputs.map(input => input.originalShape[this.concatAxis])
                                                           .reduce((i, j) => i + j);
        this.output.indicesForReshaped = tensorUtils.createIndicesFor2DReshaped(this.output.originalShape);
      }
    }

    if (!this.mergeProgram) {
      const outputShape = this.output.textureSlices
        ? this.output.textureSliceShape
        : this.output.textureShape
      const mergeProgramSource = contatenation(
        inputs.length,
        inputs.map(input => input.textureShape),
        outputShape
      );
      this.mergeProgram = webgl2.createProgram(mergeProgramSource);
    }
    webgl2.runProgram({
      program: this.mergeProgram,
      output: this.output,
      inputs: inputs.map((input, i) => ({ input, name: `inputs[${i}]` })),
      supportSliceTexture: true
    });
    return this.output;
  }
}
