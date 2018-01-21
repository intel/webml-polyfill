import {NeuralNetworkContext} from './nn'

class WebMLPolyfill {
	constructor() {
    this._nnContext;
  }

  getNeuralNetworkContext() {
    if (typeof this._nnContext === 'undefined') {
      this._nnContext = new NeuralNetworkContext();
    }
    return this._nnContext;
  }
}

if (typeof navigator.ml === 'undefined') {
  navigator.ml = new WebMLPolyfill();
}
