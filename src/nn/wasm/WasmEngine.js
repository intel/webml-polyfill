import Module from './nn_ops'

var engine = null;
export default class WasmEngine {
  static getInstance() {
    return new Promise(resolve => {
      if (engine === null) {
        Module().then(module => {
          // https://github.com/kripken/emscripten/issues/5820#issuecomment-353605456
          delete module['then'];
          engine = module;
          resolve(engine);
        });
      } else {
        resolve(engine);
      }
    });
  }
}