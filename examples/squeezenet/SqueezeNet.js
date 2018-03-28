let nnPolyfill, nnNative;
if (navigator.ml.isPolyfill) {
  nnNative = null;
  nnPolyfill = navigator.ml.getNeuralNetworkContext();
} else {
  nnNative = navigator.ml.getNeuralNetworkContext();
  nnPolyfill = navigator.ml_polyfill.getNeuralNetworkContext();
}

class SqueezeNet {
  constructor(onnxModel, backend) {
    this._onnxModel = onnxModel;
    this._model = null;
    this._compilation;
    this._execution;
    this._tensorIds = [];
    this._operandIndex = 0;
    if (typeof backend !== 'undefined') {
      this._backend = backend;
    } else {
      if (nnNative) {
        this._backend = 'WebML';
      } else {
        this._backend = 'WASM';
      }
    }
    if (this._backend === 'WebML') {
      if (nnNative === null) {
        throw Error('Fails to initialize neural network context');
      }
      this._nn = nnNative;
    } else if (this._backend === 'WASM' || this._backend === 'WebGL2') {
      this._nn = nnPolyfill;
    }
  }

  async createCompiledModel() {
    let options = {};
    if (this._backend === 'WebGL2') {
      options.useWebGL2 = true;
    }
    this._model = await this._nn.createModel(options);

    this._addTensorOperands();
    this._addOpsAndParams();

    await this._model.finish();
    this._compilation = await this._model.createCompilation();
    this._compilation.setPreference(this._nn.PREFER_FAST_SINGLE_ANSWER);
    await this._compilation.finish();
    this._execution = await this._compilation.createExecution();
  }

  async compute(inputTensor, outputTensor) {
    this._execution.setInput(0, inputTensor);
    this._execution.setOutput(0, outputTensor);

    let error = await this._execution.startCompute();
    if (error) {
      return error;
    }
    return 'success';
  }

  _addTensorOperands() {
    throw Error('Not implemented');
  }

  _addOpsAndParams() {
    throw Error('Not implemented');
  }
}