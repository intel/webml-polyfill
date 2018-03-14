import {OperationCodeToLayersMap, OperationCodeAttrsMap, WebGL2SpecialLayers} from './utils/modelUtils'
import Tensor from './Tensor'

/**
 * WebGL2 Model class
 */
export default class Model {
  /**
   * Create WebGL2 Model class in nn/Model.js
   *
   * @param {Object} model - Model from nn/Model.js
   */
  constructor(model) {
    this._model = model;
    this._operands = model._operands;
    this._layers = [];
    this.supportTopClasses = false; 

  }

  /**
   * Called in nn/Compilation.js
   *
   */
  prepareModel() {
    return new Promise((resolve) => {
      this._layers.push(new WebGL2SpecialLayers.Input());
      this._model._operations.forEach(op => {
        let attrs = OperationCodeAttrsMap.get(op.type)(this._operands, op.inputs, op.output);
        let LayerClass = OperationCodeToLayersMap.get(op.type);
        this._layers.push(new LayerClass(attrs));
      });
      if (this.supportTopClasses) {
        this._layers.push(new WebGL2SpecialLayers.TopClasses({ numTopClasses: 3 }));
      }
      // console.log(this._layers);
      resolve('compile success');
    });
  }

  /**
   * Called in nn/Execution.js
   *
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  execute(inputs, outputs) {
    return new Promise((resolve) => {
      let nnOperands = this._operands;      
      let inputBuffer = inputs[0].buffer;
      let inputIndex = inputs[0].index;
      let inputTensor = new Tensor(inputBuffer, nnOperands[inputIndex].dimensions.slice(1,4));
      let TopClassesOut;
      let output;
      this._layers.forEach((layer, i) => {
        if (i == 0) {
          output = layer.call(inputBuffer, nnOperands[inputIndex].dimensions.slice(1,4), Float32Array);
        } else if (this.supportTopClasses && i === this._layers.length - 1){
          output = layer.call(output);
        } else {
          output = layer.call(output);
        }
        // console.log(layer)
      });
      output.transferFromGLTexture();
      let outputBuffer = outputs[0].buffer;
      outputBuffer.set(output.tensor.data);
      resolve('execute success');
    });
  }
}

