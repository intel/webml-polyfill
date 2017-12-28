import * as nn from './nn'

class WebNN {
  constructor() {
    this.Compilation = nn.Compilation;
    this.Execution = nn.Execution;
    this.Memory = nn.Memory;
    this.Model = nn.Model;
  }
}

class WebMLPolyfill {
	constructor() {
    this.nn = new WebNN();
  }
}

if (typeof navigator.ml === 'undefined') {
  navigator.ml = new WebMLPolyfill();
}
