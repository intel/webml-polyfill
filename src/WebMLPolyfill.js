import * as nn from './nn'

class WebNN {
  constructor() {
    this.Compilation = nn.Compilation;
    this.Execution = nn.Execution;
    this.Model = nn.Model;
    this.OperationCode = nn.OperationCode;
    this.OperandCode = nn.OperandCode;
    this.PaddingCode = nn.PaddingCode;
    this.PreferenceCode = nn.PreferenceCode;
    this.FuseCode = nn.FuseCode;
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
