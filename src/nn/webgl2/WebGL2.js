import { vertexShaderSource } from './webgl/vertexShader/vertexShader'

/**
 * WebGL2 class
 */
class WebGL2 {
  constructor() {
    this.supportWebGL2 = false;
    this._vertexShader = null;

    this.toDelete = { 
      textures: [], 
      buffers: [],
      shaders: [],
      programs: [],
      framebuffers: []
    };

    this.canvas = document.createElement('canvas');
    this.context = this.canvas.getContext('webgl2');
    const gl = this.context;
    if (gl) {
      this.supportWebGL2 = true;

      // gl.R32F sized format become color-renderable
      gl.getExtension('EXT_color_buffer_float'); 

      this.MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      this.MAX_TEXTURE_IMAGE_UNITS = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
      this._loadVertexShader();
    } else {
      console.log('[WebGL2] Do not support WebGL2.');
    }
  }

    /**
   * Load vertex shader.
   */
  _loadVertexShader() {
    const gl = this.context;

    let vertexShader = gl.createShader(gl.VERTEX_SHADER);
    if (vertexShader == null) {
      this.deleteAll();
      throw new Error('[WebGL2] Unable to create vertex shader');
    }
    this.toDelete.shaders.push(vertexShader);

    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);

    let compiled = gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS);
    if (!compiled) {
      const errorMsg = '[WebGL2] Failed to compile vertex shader: ' + gl.getShaderInfoLog(vertexShader);
      this.deleteAll();
      throw new Error(errorMsg);
    }

    this._vertexShader = vertexShader;
  }

  /**
   * initialize Vertex Buffers
   *
   * @param {WebGLProgram} program
   */
  _initVertexBuffers(program) {
    const gl = this.context;

    let position = gl.getAttribLocation(program, 'position');
    if (position < 0) {
      this.deleteAll();
      throw new Error('[WebGL2] Failed to get position in vertexShaderSource');
    }

    let vertexPositionBuffer = gl.createBuffer();
    if (!vertexPositionBuffer) {
      this.deleteAll();
      throw new Error('[WebGL2] Failed to create the buffer object');
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexPositionBuffer);
    this.toDelete.buffers.push(vertexPositionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0]),
      gl.STATIC_DRAW
    );
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(position);

    let texcoord = gl.getAttribLocation(program, 'texcoord');
    if (texcoord < 0) {
      this.deleteAll();
      throw new Error('[WebGL2] Failed to get texcoord in vertexShaderSource');
    }
    
    let vertexTexcoordBuffer = gl.createBuffer();
    if (!vertexTexcoordBuffer) {
      this.deleteAll();
      throw new Error('[WebGL2] Failed to create the buffer object');
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, vertexTexcoordBuffer);
    this.toDelete.buffers.push(vertexTexcoordBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER, 
      new Float32Array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]), 
      gl.STATIC_DRAW
    );
    gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(texcoord);
  }

  /**
   * Load fragment shader, creates program and initialize Vertex Buffers
   *
   * @param {string} fragmentShaderSource
   * @returns {WebGLProgram} program
   */
  createProgram(fragmentShaderSource) {
    const gl = this.context;

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (fragmentShader == null) {
      this.deleteAll();
      throw new Error('[WebGL2] Unable to create fragment shader');
    }
    this.toDelete.shaders.push(fragmentShader);

    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);

    let compiled = gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS);
    if (!compiled) {
      const errorMsg = '[WebGL2] Failed to compile fragment shader: ' + gl.getShaderInfoLog(fragmentShader);
      this.deleteAll();
      throw new Error(errorMsg);
    }

    const program = gl.createProgram();
    if (!program) {
      this.deleteAll();
      throw new Error('[WebGL2] Unable to create program');
    }
    this.toDelete.programs.push(program);

    gl.attachShader(program, this._vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    let linked = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (!linked) {
      const errorMsg = '[WebGL2] Failed to link program: ' + gl.getProgramInfoLog(program);
      this.deleteAll();
      throw new Error(errorMsg);
    }

    this._initVertexBuffers(program);
    return program;
  }

  /**
   * Runs program
   *
   * @param {WebGLProgram} options.program
   * @param {Tensor} options.output
   * @param {Object[]} options.inputs For example: [x, indexMap, kernel, bias]
   * @param {Object[]} options.uniforms
   * @param {Boolean} supportSliceTexture
   */
  runProgram({ program, output, inputs, uniforms, supportSliceTexture = false }) {
    if (!program) throw new Error('[WebGL2] No program in WebGL2 runProgram');
    if (!output) throw new Error('[WebGL2] No output in WebGL2 runProgram');
    if (!inputs) throw new Error('[WebGL2] No inputs in WebGL2 runProgram');

    const gl = this.context;

    gl.useProgram(program);
    if (uniforms && Array.isArray(uniforms)) {
      this._bindUniforms(program, uniforms);
    }

    if (output.textureSlices) {
      if (!supportSliceTexture) {
        throw new Error('[WebGL2] Program does not support texture fragments');
      }

      // Get inputs with slices such as indexMap, but not x(colStackTexture) or kernel/bias(no slices)
      let inputsWithSlices = inputs.filter(
        obj => obj.input.textureSlices && !obj.input.colStackTexture
      );
      let numSlices = output.textureSlices.length;
      if (inputsWithSlices.some(obj => obj.input.textureSlices.length !== numSlices)) {
        throw new Error('[WebGL2] Number of texture slices in inputs and output do not match');
      }

      // For each k, run 1 slice
      for (let k = 0; k < numSlices; k++) {
        this.bindOutputTexture(output.textureSlices[k], output.textureSliceShape);
        this._bindInputTextures(program, inputs, k);
        gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
      }
    } else {
      this.bindOutputTexture(output.texture, output.textureShape);
      this._bindInputTextures(program, inputs);
      gl.drawArrays(gl.TRIANGLE_FAN, 0, 4);
    }
  }   

    /**
   * Bind uniforms
   *
   * @param {WebGLProgram} program
   * @param {Object[]} uniforms
   */
  _bindUniforms(program, uniforms) {
    const gl = this.context;

    uniforms.forEach(({ value, type, name }) => {
      let uniformLocation = gl.getUniformLocation(program, name);
      if (type === 'float') {
        gl.uniform1f(uniformLocation, value);
      } else if (type === 'int' || type === 'bool') {
        gl.uniform1i(uniformLocation, value);
      }
    })
  }

  /**
   * Bind input textures
   *
   * @param {WebGLProgram} program
   * @param {Object[]} inputs
   * @param {number} [k]
   */
  _bindInputTextures(program, inputs, k) {
    const gl = this.context;

    inputs.forEach(({ input, name }, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      if (input.textureSlices) {
        if (input.colStackTexture) {
          const { textureTarget } = this.getTextureOptions(input.textureType, input.textureFormat);
          gl.bindTexture(textureTarget, input.colStackTexture);
        } else {
          const { textureTarget } = this.getTextureOptions(input.textureType, input.textureFormat);
          gl.bindTexture(textureTarget, input.textureSlices[k]);
        }
      } else {
        const { textureTarget } = this.getTextureOptions(input.textureType, input.textureFormat);
        gl.bindTexture(textureTarget, input.texture);
      }
      gl.uniform1i(gl.getUniformLocation(program, name), i);
    })
  }

  /**
   * Bind output texture
   *
   * @param {WebGLTexture} outputTexture
   * @param {number[]} shape
   */
  bindOutputTexture(outputTexture, shape) {
    const gl = this.context;

    gl.viewport(0, 0, shape[1], shape[0]);
    if (!this.framebuffer) {
      this.framebuffer = gl.createFramebuffer();
      this.toDelete.framebuffers.push(this.framebuffer);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
  }

  /**
   * Reads pixel data from framebuffer
   *
   * @param {number[]} shape
   * @returns {Float32Array}
   */
  readData(shape) {
    const gl = this.context;

    let buf = new ArrayBuffer(shape[0] * shape[1] * 4 * 4);
    let view = new Float32Array(buf);
    gl.readPixels(0, 0, shape[1], shape[0], gl.RGBA, gl.FLOAT, view);
    let out = [];
    for (let i = 0; i < view.length; i += 4) {
      out.push(view[i]);
    }
    return new Float32Array(out);
  }

  /**
   * Gets WebGLTexture options constants
   */
  getTextureOptions(type, format) {
    const gl = this.context;

    const targetMap = {
      '2d': gl.TEXTURE_2D
    };

    const internalFormatMap = {
      float: gl.R32F,
      int: gl.R32I
    };

    const formatMap = {
      float: gl.RED,
      int: gl.RED_INTEGER
    };

    const typeMap = {
      float: gl.FLOAT,
      int: gl.INT
    };

    let textureTarget = targetMap[type];
    let textureInternalFormat = internalFormatMap[format];
    let textureFormat = formatMap[format];
    let textureType = typeMap[format];

    return { textureTarget, textureInternalFormat, textureFormat, textureType }
  }

  /**
   * Deletes all stored references to WebGL textures, buffers, shaders, programs and framebuffers
   */
  deleteAll() {
    const gl = this.context;

    this.toDelete.textures.forEach(texture => gl.deleteTexture(texture));
    this.toDelete.buffers.forEach(buffer => gl.deleteBuffer(buffer));
    this.toDelete.shaders.forEach(shader => gl.deleteShader(shader));
    this.toDelete.programs.forEach(program => gl.deleteProgram(program));
    this.toDelete.framebuffers.forEach(Framebuffer => gl.deleteFramebuffer(Framebuffer));
    this.framebuffer = null;
    this.readFramebuffer = null;
    this.concateFramebuffer = null;

    this.toDelete = { 
      textures: [], 
      buffers: [],
      shaders: [],
      programs: [],
      framebuffers: []
    };
  }
}

const webgl2 = new WebGL2();
// webgl2.MAX_TEXTURE_SIZE = 4096;
// webgl2.MAX_TEXTURE_IMAGE_UNITS = 16;
// console.log(`[WebGL2] MAX_TEXTURE_SIZE: ${webgl2.MAX_TEXTURE_SIZE}`)
// console.log(`[WebGL2] MAX_TEXTURE_IMAGE_UNITS: ${webgl2.MAX_TEXTURE_IMAGE_UNITS}`)

export default webgl2;