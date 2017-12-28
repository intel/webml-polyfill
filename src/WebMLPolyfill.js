class WebNN {
  constructor() {
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
