class WebGLUtils {
  constructor(context) {

    this._gl = context;
    this._fbo = {};
    this._tex = {};

  }

  setTexture(name, tex) {
    this._tex[name] = tex;
  }

  getTexture(name) {
    return this._tex[name];
  }

  setup2dQuad() {
    const quad = new Float32Array([-1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1]);
    const vbo = this._gl.createBuffer();
    this._gl.bindBuffer(this._gl.ARRAY_BUFFER, vbo);
    this._gl.bufferData(this._gl.ARRAY_BUFFER, quad, this._gl.STATIC_DRAW);
    this._gl.enableVertexAttribArray(0);
    this._gl.vertexAttribPointer(0, 2, this._gl.FLOAT, false, 0, 0);
  }

  createAndBindTexture(params) {

    const name = params.name;
    const filter = params.filter || this._gl.LINEAR;
    const target = params.target || this._gl.TEXTURE_2D;
    const level = params.level || 0;
    const width = params.width || 1;
    const height = params.height || 1;
    const border = params.border || 0;
    const format = params.format || this._gl.RGBA;
    const internalformat = params.internalformat || format;
    const type = params.type || this._gl.UNSIGNED_BYTE;
    const source = params.format || null;

    const texture = this._gl.createTexture();

    this._gl.bindTexture(
      this._gl.TEXTURE_2D,
      texture
    );
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

    this._tex[name] = texture;

    return texture;
  }

  createTextures(textures) {
    for (let tex of textures) {
      this.createAndBindTexture(tex);
    }
  }

  bindInputTextures(textures) {
    for (let i = 0; i < textures.length; i++) {
      this._gl.activeTexture(this._gl.TEXTURE0 + i);
      this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex[textures[i]]);
    }
  }

  bindFramebuffer(fboName) {
    if (fboName === null) {
      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
    } else {
      this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo[fboName]);
    }
  }

  bindTexture(texName) {
    this._gl.bindTexture(this._gl.TEXTURE_2D, this._tex[texName]);
  }

  createTexInFrameBuffer(fboName, texturesParams) {

    const newFbo = this._gl.createFramebuffer();
    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, newFbo);
    this._fbo[fboName] = newFbo;

    const attachments = [];

    for (let i = 0; i < texturesParams.length; i++) {

      const params = texturesParams[i];
      const attach = (params.attach || i) + this._gl.COLOR_ATTACHMENT0;
      attachments.push(attach);

      const newTex = this.createAndBindTexture(params);

      this._gl.framebufferTexture2D(
        this._gl.FRAMEBUFFER,
        attach,
        this._gl.TEXTURE_2D,
        newTex,
        0
      );
    }

    if (attachments.length > 1) {
      this._gl.drawBuffers(attachments);
    }

    let status = this._gl.checkFramebufferStatus(this._gl.FRAMEBUFFER);
    if (status !== this._gl.FRAMEBUFFER_COMPLETE) {
      console.warn('FBO is not complete');
    }

    this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
  }

  setViewport(width, height) {
    this._gl.viewport(0, 0, width, height);
  }

  render() {
    this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
  }

  _toFastProperties(obj) { // force %ToFastProperties
    function dummy() {
      this.x = 0;
    }
    dummy.prototype = obj;
    new dummy();
    new dummy();
  }

  freeze() {
    if (window.chrome) {
      this._toFastProperties(this._tex);
      this._toFastProperties(this._fbo);
    }
  }

  delete() {
    this._fbo = {};
    this._tex = {};
  }
}