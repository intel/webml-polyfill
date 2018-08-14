if (!global._babelPolyfill) {
	require('babel-polyfill');
}

import Module from './nn_ops'

var nn_ops = null;
export default async function getNNOpsInstance() {
  return new Promise(resolve => {
    if (nn_ops === null) {
      Module().then(m => {
        // https://github.com/kripken/emscripten/issues/5820#issuecomment-353605456
        delete m['then'];
        nn_ops = m;
        resolve(nn_ops);
      });
    } else {
      resolve(nn_ops);
    }
  });
}