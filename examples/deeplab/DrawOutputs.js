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

    this._segMap = null;
    this._predictions = null;
    this._scaledShape = [224, 224];

    this._effect = 'label';
    this._zoom = 2;
    this._bgColor = [57, 135, 189]; // rgb
    this._blurRadius = 0;
    this._colorMapAlpha = 0.7;
    this._halfKernel = [1];
  
    this._correctionFactor = 0.99;

    this._colorPalette = new Uint8Array([
      45, 52, 54,
      85, 239, 196,
      129, 236, 236,
      116, 185, 255,
      162, 155, 254,
      223, 230, 233,
      0, 184, 148,
      0, 206, 201,
      9, 132, 227,
      39, 60, 117,
      108, 92, 231,
      178, 190, 195,
      255, 234, 167,
      250, 177, 160,
      255, 118, 117,
      253, 121, 168,
      99, 110, 114,
      253, 203, 110,
      225, 112, 85,
      214, 48, 49,
      232, 67, 147,
    ]);
  }

  get zoom() {
    return this._zoom;
  }

  set zoom(val) {
    this._zoom = val;

    // all FRAMEBUFFERs should be re-setup when zooming
    this.setup().then(_ => this.drawOutputs(this._segMap));
  }

  get bgColor() {
    return this._bgColor;
  }

  set bgColor(rgb) {
    this._bgColor = rgb;
    this.drawOutputs();
  }

  get blurRadius() {
    return this._blurRadius;
  }

  set blurRadius(radius) {
    if (this._blurRadius === radius)
      return;

    this._blurRadius = radius;
    let kernelSize = radius * 2 + 1;
    let kernel1D = this._generateGaussianKernel1D(kernelSize, 25);
    // take the second half
    this._halfKernel = kernel1D.slice(radius);

    // re-setup shader with new kernel
    this.setup().then(_ => this.drawOutputs());
  }

  get colorMapAlpha() {
    return this._colorMapAlpha;
  }

  set colorMapAlpha(alpha) {
    if (this._colorMapAlpha === alpha)
      return;

    this._colorMapAlpha = alpha;

    // re-setup shader with new alpha
    this.setup().then(_ => this.drawOutputs());
  }

  get effect() {
    return this._effect;
  }

  set effect(val) {
    this._effect = val;

    // shaders need to be rewired when changing effect
    this.setup().then(_ => this.drawOutputs(this._segMap));
  }

  async setup() {

    //
    // +------+                     +----+
    // |  im  |--.               .->| fg |-------------------.
    // +------+  |  ===========  |  +----+  ===============  |  ==========  
    //           |->| extract |--|          | blur shader |  |->| blend  |-> out
    // +------+  |  | shader  |  |  +----+  |     or      |  |  | shader |
    // | mask |--'  ===========  '->| bg |->| fill shader |--'  ==========
    // +------+                     +----+  ===============      
    //

    this._setupQuad();
    if (this._effect === 'label') {
      this._setupColorizeShader();
    } else {
      this._setupExtractShader();

      if (this._effect === 'blur') {
        this._setupBlurShader();
      } else if (this._effect === 'fill') {
        this._setupFillShader();
      }

      this._setupBlendShader();
    }
  }

  _setupQuad() {
    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1]);
    const vbo = this._gl.createBuffer();
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, vbo);
    this._gl.bufferData(this._gl.ARRAY_BUFFER, quad, this._gl.STATIC_DRAW);
    this._gl.enableVertexAttribArray(0);
    this._gl.vertexAttribPointer(0, 2, this._gl.FLOAT, false, 0, 0);
  }

  _setupColorizeShader() {
    const colorizeShaderVs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, -0.5) + 0.5;
      }`;

    const colorizeShaderFs =
      `#version 300 es
      precision highp float;
      out vec4 out_color;

      uniform sampler2D u_image;
      uniform sampler2D u_predictions;
      uniform sampler2D u_palette;
      uniform int u_length;
      uniform float u_alpha;

      in vec2 v_texcoord;

      void main()
      {
        vec2 correct_cord = v_texcoord * vec2(${this._correctionFactor}, ${this._correctionFactor});
        float label = texture(u_predictions, correct_cord).a * 255.0;
        vec4 label_color = texture(u_palette, vec2((label + 0.5) / float(u_length), 0.5));
        vec4 im_color = texture(u_image, v_texcoord);
        out_color = mix(im_color, label_color, u_alpha);
      }`;

    this._shader.colorize = new Shader(this._gl, colorizeShaderVs, colorizeShaderFs);
    this._shader.colorize.use();
    this._shader.colorize.setUniform1i('u_image', 0);   // texture units 0
    this._shader.colorize.setUniform1i('u_predictions', 1);    // texture units 1
    this._shader.colorize.setUniform1i('u_palette', 2); // texture units 2
    this._shader.colorize.setUniform1f('u_alpha', this._colorMapAlpha);
    this._shader.colorize.setUniform1i('u_length', this._colorPalette.length / 3);


    this._tex.palette = this._createAndBindTexture(this._gl.NEAREST);
    this._gl.texImage2D(
      this._gl.TEXTURE_2D,
      0,
      this._gl.RGB,
      this._colorPalette.length / 3,
      1,
      0,
      this._gl.RGB,
      this._gl.UNSIGNED_BYTE,
      this._colorPalette
    );
  }

  _setupExtractShader() {

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
        vec2 correct_cord = v_texcoord * vec2(${this._correctionFactor}, ${this._correctionFactor});
        float bg_alpha = texture(u_mask, correct_cord).a;
        float fg_alpha = 1.0 - bg_alpha;

        fg_color = vec4(texture(u_image, v_texcoord).xyz * fg_alpha, fg_alpha);
        bg_color = vec4(texture(u_image, v_texcoord).xyz * bg_alpha, bg_alpha);
      }`;

    this._shader.extract = new Shader(this._gl, extractShaderVs, extractShaderFs);
    this._createTexInFrameBuffer('extract',
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
  }

  _setupBlurShader() {

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
      uniform int raidius;
      uniform float kernel[${this._blurRadius + 1}];

      // https://learnopengl.com/code_viewer_gh.php?code=src/5.advanced_lighting/7.bloom/7.blur.fs

      void main()
      {             
        vec2 tex_offset = 1.0 / vec2(textureSize(bg, 0)); // gets size of single texel
        vec4 bg_color = texture(bg, v_texcoord);
        vec4 result = bg_color * kernel[0];
        if (first_pass) {
          for (int i = 1; i < ${this._blurRadius + 1}; ++i) {
            result += texture(bg, v_texcoord + vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
            result += texture(bg, v_texcoord - vec2(tex_offset.x * float(i), 0.0)) * kernel[i];
          }
        } else {
          for (int i = 1; i < ${this._blurRadius + 1}; ++i) {
            result += texture(bg, v_texcoord + vec2(0.0, tex_offset.y * float(i))) * kernel[i];
            result += texture(bg, v_texcoord - vec2(0.0, tex_offset.y * float(i))) * kernel[i];
          }
        }
        out_color = result;
      }`;

    this._shader.blur = new Shader(this._gl, blurShaderVs, blurShaderFs);
    this._createTexInFrameBuffer('blurFirstPassResult',
      [{
        texName: 'blurFirstPassResult',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );

    this._createTexInFrameBuffer('styledBg',
      [{
        texName: 'styledBg',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );
  }

  _setupFillShader() {
    const fillShaderVs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      
      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
      }`;

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

    this._shader.fill = new Shader(this._gl, fillShaderVs, fillShaderFs);
    this._createTexInFrameBuffer('styledBg',
      [{
        texName: 'styledBg',
        width: this._scaledShape[0] * this._zoom,
        height: this._scaledShape[1] * this._zoom,
      }]
    );
  }

  _setupBlendShader() {
    const blendShaderVs =
      `#version 300 es
      in vec4 a_pos;
      out vec2 v_texcoord;
      
      void main() {
        gl_Position = a_pos;
        v_texcoord = a_pos.xy * vec2(0.5, 0.5) + 0.5;
      }`;

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

    this._shader.blend = new Shader(this._gl, blendShaderVs, blendShaderFs);
    this._shader.blend.use();
    this._shader.blend.setUniform1i('fg', 0); // texture units 0
    this._shader.blend.setUniform1i('bg', 1); // texture units 1
    this._shader.blend.setUniform1i('orig', 2); // texture units 2
  }

  uploadNewTexture(imageSource, scaledShape) {
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

  async drawOutputs(newSegMap) {

    let start = performance.now();

    if (typeof newSegMap !== 'undefined') {
      this._segMap = newSegMap;
      let numOutputClasses = newSegMap.outputShape[2];

      if (this.effect === 'label') {
        this._predictions = this._argmax(newSegMap.data, numOutputClasses);
      } else {
        // convert precdictions to a binary mask
        this._predictions = this._argmaxPerson(newSegMap.data, numOutputClasses);
      }

      const scaledWidth = this._scaledShape[0];
      const scaledHeight = this._scaledShape[1];
      const outputWidth = this._segMap.outputShape[0];
      const outputHeight = this._segMap.outputShape[1];
      const isScaled = outputWidth === scaledWidth || outputHeight === scaledHeight;

      this._gl.canvas.width = scaledWidth * this._zoom;
      this._gl.canvas.height = scaledHeight * this._zoom;
      this._gl.viewport(
        0,
        0,
        this._gl.drawingBufferWidth,
        this._gl.drawingBufferHeight
      );

      // upload predictions(mask) texture
      this._tex.predictions = this._createAndBindTexture(this._gl.NEAREST);
      this._gl.texImage2D(
        this._gl.TEXTURE_2D,
        0,
        this._gl.ALPHA,
        isScaled ? scaledWidth : outputWidth,
        isScaled ? scaledHeight : outputHeight,
        0,
        this._gl.ALPHA,
        this._gl.UNSIGNED_BYTE,
        this._predictions
      );
      this._gl.pixelStorei(this._gl.UNPACK_ALIGNMENT, 1);
    }

    if (this._effect === 'label') {
      this._drawColorLabel();
    } else {
      this._drawPerson();
    }

    let elapsed = performance.now() - start;
    console.log(`Draw time: ${elapsed.toFixed(2)} ms`);
    return elapsed;
  }

  _drawColorLabel() {
    this._shader.colorize.use();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    this._gl.activeTexture(this._gl.TEXTURE0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.image);
    this._gl.activeTexture(this._gl.TEXTURE1);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.predictions);
    this._gl.activeTexture(this._gl.TEXTURE2);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.palette);
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);

    // generate label map. { labelName: [ labelName, rgbTuple ] }
    let uniqueLabels = new Set(this._predictions);
    let labelMap = {};
    for (let labelId of uniqueLabels) {
      let labelName = this._segMap.labels[labelId];
      let rgbTuple = this._colorPalette.slice(labelId*3, (labelId+1)*3);
      labelMap[labelId] = [labelName, rgbTuple];
    }
    this._showLegends(labelMap);
    return labelMap;
  }

  _showLegends(labelMap) {
    $('.labels-wrapper').empty();
    for (let id in labelMap) {
      let labelDiv = $(`<div class="col-12 seg-label" data-label-id="${id}"/>`)
        .append($(`<span style="color:rgb(${labelMap[id][1]})">⬤</span>`))
        .append(`${labelMap[id][0]}`);
      $('.labels-wrapper').append(labelDiv);
    }
  }

  highlightHoverLabel(hoverPos) {
    if (hoverPos === null) {
      // clear highlight when mouse leaving canvas
      $('.seg-label').removeClass('highlight');
      return;
    }

    let outputW = this._segMap.outputShape[0];
    let x = Math.floor(this._correctionFactor * hoverPos.x / this._zoom);
    let y = Math.floor(this._correctionFactor * hoverPos.y / this._zoom);
    let labelId = this._predictions[x + y * outputW];

    $('.seg-label').removeClass('highlight');
    $('.labels-wrapper').find(`[data-label-id="${labelId}"]`).addClass('highlight');
  }

  _drawPerson() {

    // feed image and mask into extract shader
    this._shader.extract.use();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.extract);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    this._gl.activeTexture(this._gl.TEXTURE0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.image);
    this._gl.activeTexture(this._gl.TEXTURE1);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.predictions);
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);


    if (this._effect == 'blur') {
      // feed extracted background into blur shader
      this._shader.blur.use();
      this._shader.blur.setUniform1fv('kernel', this._halfKernel);

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
      let fillColor = this._bgColor.map(x => x / 255);
      this._shader.fill.use();
      this._shader.fill.setUniform4f('fill_color', ...fillColor, 1);
      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo.styledBg);
      this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
      this._gl.activeTexture(this._gl.TEXTURE0);
      this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.bg);
      this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
    }

    // feed into blend shader
    this._shader.blend.use();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
    this._gl.clear(this._gl.COLOR_BUFFER_BIT | this._gl.DEPTH_BUFFER_BIT);
    this._gl.activeTexture(this._gl.TEXTURE0);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.fg);
    this._gl.activeTexture(this._gl.TEXTURE1);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.styledBg);
    this._gl.activeTexture(this._gl.TEXTURE2);
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex.image);
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
  }

  _argmax(array, span) {
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
      mask[i] = maxIdx;
    }
    return mask;
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

  _generateGaussianKernel1D(kernelSize, sigma = 1) {
    const gaussian = (x, sigma) =>
      1 / (Math.sqrt(2 * Math.PI) * sigma) * Math.exp(-x * x / (2 * sigma * sigma));
    const kernel = [];
    const radius = (kernelSize - 1) / 2;
    for (let x = -radius; x <= radius; x++) {
      kernel.push(gaussian(x, sigma));
    }

    // normalize kernel
    const sum = kernel.reduce((x, y) => x + y);
    return kernel.map((x) => x / sum);
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

  _createTexInFrameBuffer(fboName, texturesConfig) {

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

  setUniform1f(name, x) {
    this._gl.uniform1f(this._gl.getUniformLocation(this._prog, name), x);
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