class Shader {
  constructor(gl, vertShaderSrc, fragShaderSrc) {
    this._gl = gl;
    this._loc = [];

    const vs = this._createShader(this._gl.VERTEX_SHADER, vertShaderSrc);
    const fs = this._createShader(this._gl.FRAGMENT_SHADER, fragShaderSrc);
    this._prog = this._createProgram(vs, fs);

    // get location of all uniforms in vertex and fragment shader codes
    // note: this regex will take into account uniforms within comments
    const regex = /uniform\s+[^\s]+\s+([_a-zA-Z][_a-zA-Z0-9]*)[=\[\s]?[^;]*;/g;
    const src = vertShaderSrc + fragShaderSrc;
    for (let match; match = regex.exec(src); ) {
      let uniform = match[1];
      // location will be null if uniform is commented or optimized out 
      this._loc[uniform] = this._gl.getUniformLocation(this._prog, uniform);
    }
  }

  use() {
    this._gl.useProgram(this._prog);
  }

  set1i(name, x) {
    this._gl.uniform1i(this._loc[name], x);
  }

  set4f(name, w, x, y, z) {
    this._gl.uniform4f(this._loc[name], w, x, y, z);
  }

  set2f(name, x, y) {
    this._gl.uniform2f(this._loc[name], x, y);
  }

  set1f(name, x) {
    this._gl.uniform1f(this._loc[name], x);
  }

  set1fv(name, arr) {
    this._gl.uniform1fv(this._loc[name], arr);
  }

  _createShader(type, source) {
    let shader = this._gl.createShader(type);
    this._gl.shaderSource(shader, source);
    this._gl.compileShader(shader);
    if (!this._gl.getShaderParameter(shader, this._gl.COMPILE_STATUS)) {
      let log = this._gl.getShaderInfoLog(shader);
      this._gl.deleteShader(shader);
      throw new Error(log);
    }
    return shader;
  }

  _createProgram(vertexShader, fragmentShader) {
    let program = this._gl.createProgram();
    this._gl.attachShader(program, vertexShader);
    this._gl.attachShader(program, fragmentShader);
    this._gl.linkProgram(program);
    if (!this._gl.getProgramParameter(program, this._gl.LINK_STATUS)) {
      let log = this._gl.getProgramInfoLog(program);
      this._gl.deleteProgram(program);
      throw new Error(log);
    }
    return program;
  }
}