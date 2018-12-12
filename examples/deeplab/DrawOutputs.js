class Renderer {
  constructor(canvas) {

    this._gl = canvas.getContext('webgl2');
    if (this._gl === null) {
      throw new Error('Unable to initialize WebGL.');
    }

    this._program = null;
    this._tex = {};
    this._fbo = {};
    this._shader = {};

    this._mask = null;
    this._segMap = null;
    this._scaledShape = [224, 224];

    this._effect = 'fill';
    this._zoom = 1;
    this._bgcolor = [57, 135, 189]; // rgb
    this._blurRadius = 1;
  }

  get zoom() {
    return this._zoom;
  }

  set zoom(val) {
    this._zoom = val;

    // all FRAMEBUFFERs should be re-setup when zooming
    this.setup();
    this.drawOutputs(this._segMap);
  }

  get bgcolor() {
    return this._bgcolor;
  }

  set bgcolor(rgb) {
    this._bgcolor = rgb;
    this.drawOutputs();
  }

  get blurRadius() {
    return this._blurRadius;
  }

  set blurRadius(val) {
    this._blurRadius = val;
    this.drawOutputs();
  }

  get effect() {
    return this._effect;
  }

  set effect(val) {
    this._effect = val;
    this.drawOutputs();
  }

  setup() {

    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1]);
    const vbo = this._gl.createBuffer();
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, vbo);
    this._gl.bufferData(this._gl.ARRAY_BUFFER, quad, this._gl.STATIC_DRAW);
    this._gl.enableVertexAttribArray(0);
    this._gl.vertexAttribPointer(0, 2, this._gl.FLOAT, false, 0, 0);


    //
    // +------+                     +----+
    // |  im  |--.               .->| fg |-------------------.
    // +------+  |  ===========  |  +----+  ===============  |  ==========  
    //           |->| extract |--|          | blur shader |  |->| blend  |-> out
    // +------+  |  | shader  |  |  +----+  |     or      |  |  | shader |
    // | mask |--'  ===========  '->| bg |->| fill shader |--'  ==========
    // +------+                     +----+  ===============      
    //

    const extractShaderVs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
      }`;
    const extractShaderFs =
      `#version 300 es
      precision highp float;
      layout(location = 0) out vec4 fg_color;
      layout(location = 1) out vec4 bg_color;

      uniform sampler2D u_mask;
      uniform sampler2D u_image;

      in vec2 v_texcoord;

      void main()
      {
        float bg_alpha = texture(u_mask, v_texcoord * vec2(0.99, 0.99)).a;
        float fg_alpha = 1.0 - bg_alpha;

        fg_color = vec4(texture(u_image, v_texcoord).xyz * fg_alpha, fg_alpha);
        bg_color = vec4(texture(u_image, v_texcoord).xyz * bg_alpha, bg_alpha);
      }`;

    const blurShaderVs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      
      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
      }`;
    const blurShaderFs =
      `#version 300 es
      precision highp float;
      out vec4 out_color;
      in vec2 v_texcoord;

      uniform sampler2D bg;
      uniform bool first_pass;
      uniform float kernel[5];
      
      // https://learnopengl.com/code_viewer_gh.php?code=src/5.advanced_lighting/7.bloom/7.blur.fs

      void main()
      {             
        vec2 tex_offset = 1.0 / vec2(textureSize(bg, 0)); // gets size of single texel
        vec4 bg_color = texture(bg, v_texcoord);
        vec4 result = bg_color * kernel[0];
        if (first_pass) {
          for (int i = 1; i < 5; ++i) {
            result += texture(bg, v_texcoord + vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
            result += texture(bg, v_texcoord - vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
          }
        } else {
          for (int i = 1; i < 5; ++i) {
            result += texture(bg, v_texcoord + vec2(0.0, tex_offset.y * float(i))) * kernel[i];
            result += texture(bg, v_texcoord - vec2(0.0, tex_offset.y * float(i))) * kernel[i];
          }
        }
        out_color = result;
      }`;

    const fillShaderVs = blurShaderVs;
    const fillShaderFs =
      `#version 300 es
      precision highp float;

      in vec2 v_texcoord;
      out vec4 out_color;
      uniform vec4 fill_color;

      uniform sampler2D bg;
      
      void main() {
        float bg_alpha = texture(bg, v_texcoord).a;
        out_color = vec4(fill_color.xyz * bg_alpha, bg_alpha);
      }`;

    const blendShaderVs = blurShaderVs;
    const blendShaderFs =
      `#version 300 es
      precision highp float;

      in vec2 v_texcoord;
      out vec4 out_color;

      uniform sampler2D fg;
      uniform sampler2D bg;
      uniform sampler2D orig;

      void main() {
        vec4 fg_color = texture(fg, v_texcoord);
        vec4 bg_color = texture(bg, v_texcoord);
        vec4 orig_color = texture(orig, vec2(v_texcoord.x, (1.0-v_texcoord.y)));
        bg_color = bg_color + (1.0 - bg_color.a) * orig_color;
        out_color = fg_color + (1.0 - fg_color.a) * bg_color;
      }`;


    // Extract shader
    this._shader.extract = new Shader(this._gl, extractShaderVs, extractShaderFs);
    this._createNewTexInFrameBuffer('extract',
      [{
        texName: 'fg',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }, {
        texName: 'bg',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );
    this._shader.extract.use();
    this._shader.extract.setUniform1i('u_image', 0); // texture units 0
    this._shader.extract.setUniform1i('u_mask', 1); // texture units 1


    // Blur shader
    this._shader.blur = new Shader(this._gl, blurShaderVs, blurShaderFs);
    this._createNewTexInFrameBuffer('blurFirstPassResult',
      [{
        texName: 'blurFirstPassResult',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );


    // Fill Shader
    this._shader.fill = new Shader(this._gl, fillShaderVs, fillShaderFs);
    this._createNewTexInFrameBuffer('styledBg',
      [{
        texName: 'styledBg',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );


    // Blend shader
    this._shader.blend = new Shader(this._gl, blendShaderVs, blendShaderFs);
    this._shader.blend.use();
    this._shader.blend.setUniform1i('fg', 0); // texture units 0
    this._shader.blend.setUniform1i('bg', 1); // texture units 1
    this._shader.blend.setUniform1i('orig', 2); // texture units 2
  }

  async uploadNewTexture(imageSource, scaledShape) {
    let id = Date.now(); // used for sync
    return new Promise(resolve => {

      if (this._scaledShape[0] !== scaledShape[0] ||
        this._scaledShape[1] !== scaledShape[1]) {
        this._scaledShape = scaledShape;

        // all FRAMEBUFFERs should be re-setup when scaledShape changes
        this.setup();
      }

      this._tex.image = this._createAndBindTexture(this._gl.LINEAR);
      this._gl.texImage2D(
        this._gl.TEXTURE_2D,
        0,
        this._gl.RGBA,
        this._gl.RGBA,
        this._gl.UNSIGNED_BYTE,
        imageSource
      );

      resolve(id);
    });
  }

  drawOutputs(newSegMap) {

    let start = performance.now();

    if (typeof newSegMap !== 'undefined') {
      this._segMap = newSegMap;
      let numOutputClasses = newSegMap.outputShape[2];
      this._mask = this._argmaxPerson(newSegMap.data, numOutputClasses);

      let scaledWidth = this._scaledShape[0];
      let scaledHeight = this._scaledShape[1];
      let outputWidth = this._segMap.outputShape[0];
      let outputHeight = this._segMap.outputShape[1];
      let isScaled = outputWidth === scaledWidth || outputHeight === scaledHeight;

      this._gl.canvas.width = scaledWidth * this._zoom;
      this._gl.canvas.height = scaledHeight * this._zoom;
      this._gl.viewport(0, 0, this._gl.drawingBufferWidth, this._gl.drawingBufferHeight);

      // upload new mask texture
      this._tex.mask = this._createAndBindTexture(this._gl.NEAREST);
      this._gl.texImage2D(
        this._gl.TEXTURE_2D,
        0,
        this._gl.ALPHA,
        isScaled ? scaledWidth : outputWidth,
        isScaled ? scaledHeight : outputHeight,
        0,
        this._gl.ALPHA,
        this._gl.UNSIGNED_BYTE,
        this._mask
      );
      this._gl.pixelStorei(this._gl.UNPACK_ALIGNMENT, 1);
    }


    // feed image and mask into extract shader
    this._shader.extract.use();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.extract);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    this._gl.activeTexture(this._gl.TEXTURE0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.image);
    this._gl.activeTexture(this._gl.TEXTURE1);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.mask);
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);


    if (this._effect == 'blur') {
      // feed extracted background into blur shader
      let kernel = [0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162];
      this._shader.blur.use();
      this._shader.blur.setUniform1fv('kernel', kernel);

      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.blurFirstPassResult);
      this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
      this._shader.blur.setUniform1i('first_pass', 1);
      this._gl.activeTexture(this._gl.TEXTURE0);
      this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.bg);
      this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);

      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.styledBg);
      this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
      this._shader.blur.setUniform1i('first_pass', 0);
      // this._gl.activeTexture(this._gl.TEXTURE0);
      this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.blurFirstPassResult);
      this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
    } else {
      // feed extracted background into fill shader
      let fillColor = this._bgcolor.map(x => x / 255);
      this._shader.fill.use();
      this._shader.fill.setUniform4f('fill_color', ...fillColor, 1);
      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.styledBg);
      this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
      this._gl.activeTexture(this._gl.TEXTURE0);
      this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.bg);
      this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
    }

    const bgTex = this.effect === 'none' ? this._tex.bg : this._tex.styledBg;

    // feed into blend shader
    this._shader.blend.use();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    this._gl.activeTexture(this._gl.TEXTURE0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.fg);
    this._gl.activeTexture(this._gl.TEXTURE1);
    this._gl.bindTexture(this._gl.TEXTURE_2D, bgTex);
    this._gl.activeTexture(this._gl.TEXTURE2);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.image);
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);

    let elapsed = performance.now() - start;
    console.log(`Draw time: ${elapsed.toFixed(2)} ms`);
    return elapsed;
  }

  _argmaxPerson(array, span) {
    const PERSON_ID = 15;
    const len = array.length / span;
    const mask = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      let maxVal = Number.MIN_SAFE_INTEGER;
      let maxIdx = 0;
      for (let j = 0; j < span; j++) {
        if (array[i * span + j] > maxVal) {
          maxVal = array[i * span + j];
          maxIdx = j;
        }
      }
      mask[i] = maxIdx === PERSON_ID ? 0 : 255;
    }
    return mask;
  }


  // WebGL helper functions
  _createAndBindTexture(filter = this._gl.NEAREST) {
    let texture = this._gl.createTexture();
    this._gl.bindTexture(this._gl.TEXTURE_2D, texture);
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_WRAP_S,
      this._gl.CLAMP_TO_EDGE
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_WRAP_T,
      this._gl.CLAMP_TO_EDGE
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_MIN_FILTER,
      filter
    );
    this._gl.texParameteri(
      this._gl.TEXTURE_2D,
      this._gl.TEXTURE_MAG_FILTER,
      filter
    );
    return texture;
  }

  _createAndBindFrameBuffer() {
    let fbo = this._gl.createFramebuffer();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, fbo);
    return fbo;
  }

  _createNewTexInFrameBuffer(fboName, texturesConfig) {

    this._fbo[fboName] = this._createAndBindFrameBuffer();
    const attachments = [];

    for (let i = 0; i < texturesConfig.length; i++) {

      const config = texturesConfig[i];
      const filter = config.filter || this._gl.LINEAR;
      const target = config.target || this._gl.TEXTURE_2D;
      const level = config.level || 0;
      const internalformat = config.internalformat || this._gl.RGBA;
      const width = config.width || 0;
      const height = config.height || 0;
      const border = config.border || 0;
      const format = config.format || this._gl.RGBA;
      const type = config.type || this._gl.UNSIGNED_BYTE;
      const source = config.format || null;
      const attach = (config.attach || i) + this._gl.COLOR_ATTACHMENT0;
      attachments.push(attach);

      const newTex = this._createAndBindTexture(filter);

      this._gl.texImage2D(
        target,
        level,
        internalformat,
        width,
        height,
        border,
        format,
        type,
        source
      );

      this._gl.framebufferTexture2D(
        this._gl.FRAMEBUFFER,
        attach,
        this._gl.TEXTURE_2D,
        newTex,
        0
      );

      this._tex[config.texName] = newTex;
    }

    if (attachments.length > 1) {
      this._gl.drawBuffers(attachments);
    }

    let status = this._gl.checkFramebufferStatus(this._gl.FRAMEBUFFER);
    if (status !== this._gl.FRAMEBUFFER_COMPLETE) {
      console.warn('FBOs are not complete');
    }

    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
  }
}


class Shader {
  constructor(gl, vertShaderSrc, fragShaderSrc) {
    this._gl = gl;
    let vs = this._createShader(this._gl.VERTEX_SHADER, vertShaderSrc);
    let fs = this._createShader(this._gl.FRAGMENT_SHADER, fragShaderSrc);
    this._prog = this._createProgram(vs, fs);
  }

  use() {
    this._gl.useProgram(this._prog);
  }

  setUniform1i(name, x) {
    this._gl.uniform1i(this._gl.getUniformLocation(this._prog, name), x);
  }

  setUniform4f(name, w, x, y, z) {
    this._gl.uniform4f(this._gl.getUniformLocation(this._prog, name), w, x, y, z);
  }

  setUniform1fv(name, arr) {
    this._gl.uniform1fv(this._gl.getUniformLocation(this._prog, name), arr);
  }

  _createShader(type, source) {
    let shader = this._gl.createShader(type);
    this._gl.shaderSource(shader, source);
    this._gl.compileShader(shader);
    if (this._gl.getShaderParameter(shader, this._gl.COMPILE_STATUS)) {
      return shader;
    }
    console.log(this._gl.getShaderInfoLog(shader));
    this._gl.deleteShader(shader);
  }

  _createProgram(vertexShader, fragmentShader) {
    let program = this._gl.createProgram();
    this._gl.attachShader(program, vertexShader);
    this._gl.attachShader(program, fragmentShader);
    this._gl.linkProgram(program);
    if (this._gl.getProgramParameter(program, this._gl.LINK_STATUS)) {
      return program;
    }
    console.log(this._gl.getProgramInfoLog(program));
    this._gl.deleteProgram(program);
  }
}