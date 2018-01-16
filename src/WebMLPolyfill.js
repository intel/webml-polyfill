import {NeuralNetwork} from './nn'

class WebMLPolyfill {
	constructor() {
    this.nn = new NeuralNetwork();
  }
}

if (typeof navigator.ml === 'undefined') {
  navigator.ml = new WebMLPolyfill();
}
