class WebMLPolyfill {
	constructor() {
    }
}

if(typeof navigator.ml === 'undefined') navigator.ml = new WebMLPolyfill();
