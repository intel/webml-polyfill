import NeuralNetworkContext from './nn/NeuralNetworkContext'
import {NeuralNetworkContext as NeuralNetworkContextV2} from './nn_v2/NeuralNetworkContext'

class WebMLPolyfill {
	constructor() {
    this._nnContext = null;
    this._nnContextV2 = null;
    this.isPolyfill = true;
  }

  getNeuralNetworkContext(option = null) {
    if (!option || option === 'v1') {
      if (!this._nnContext) {
        this._nnContext = new NeuralNetworkContext();
      }
      return this._nnContext;
    } else if (option === 'v2') {
      if (!this._nnContextV2) {
        this._nnContextV2 = new NeuralNetworkContextV2();
      }
      return this._nnContextV2;
    } else {
      throw new Error(`option ${option} is not supported`);
    }
  }
}

if (typeof navigator.ml === 'undefined') {
  navigator.ml = new WebMLPolyfill();
} else {
  navigator.ml_polyfill = new WebMLPolyfill();
}