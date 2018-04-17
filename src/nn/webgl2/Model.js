import {OperationCodeToLayersMap, OperationCodeAttrsMap, WebGL2SpecialLayers} from './utils/modelUtils'
import Tensor from './Tensor'
import ops from 'ndarray-ops'

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
    this._operands = Array(model._operands.length);
    this._layers = [];
    this.supportInputLayer = true;
    this.supportTopClasses = false; 
  }

  /**
   * Called in nn/Compilation.js
   *
   */
  prepareModel() {
    return new Promise((resolve) => {
      if (this.supportInputLayer) {
        this._layers.push(new WebGL2SpecialLayers.Input());
      }
      this._model._operations.forEach((op, i) => {
        // console.log(op);
        let attrs = OperationCodeAttrsMap.get(op.type)(this._model._operands, op.inputs, op.outputs);
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
      let nnOperands = this._model._operands;
      let nnOperations = this._model._operations
      let inputBuffer = inputs[0].buffer;
      let inputIndex = inputs[0].index;
      let outputBuffer = outputs[0].buffer;
      let outputIndex = outputs[0].index;
      // let operationStart = performance.now();
      this._layers.forEach((layer, i) => {
        // let start = performance.now();
        if (i == 0) {
          if (this.supportInputLayer) {
            this._operands[inputIndex] = layer.call(inputBuffer, nnOperands[inputIndex].dimensions.slice(1,4), Float32Array);
          } else {
            let inputTensor = new Tensor(inputBuffer, nnOperands[inputIndex].dimensions.slice(1,4));
            this._operands[layer.outputs[0]] = layer.call(inputTensor);
          }
        } else if (this.supportTopClasses && i === this._layers.length - 1){
          let outBufferAndIndex = layer.call(this._operands[outputIndex]);
          // console.log(`outBufferAndIndex: ${outBufferAndIndex}`);
          outputBuffer.fill(0);
          let bufferLength = outBufferAndIndex.length / 2;
          for (let k = 0; k < bufferLength; ++k) {
            outputBuffer[outBufferAndIndex[k + bufferLength]] = outBufferAndIndex[k];
          }
          // console.log(`outputBuffer: ${outputBuffer}`);
        } else {
          if (layer.inputs.length === 1) {
            this._operands[layer.outputs[0]] = layer.call(this._operands[layer.inputs[0]]);
          } else {
            let MutiInputs = [];
            layer.inputs.forEach(input => {
              MutiInputs.push(this._operands[input]);
            });
            this._operands[layer.outputs[0]] = layer.call(MutiInputs);
          }
          // this._operands[layer.outputs[0]].transferFromGLTexture();
        }
        // console.log(i, (performance.now() - start).toFixed(2), layer);
      });
      // let operationTime = performance.now() - operationStart;
      // console.log(`Operation Time: ${operationTime.toFixed(2)} ms`);
      if (!this.supportTopClasses) {
        this._operands[outputIndex].transferFromGLTexture();
        // console.log(outputIndex, this._operands[outputIndex]);
        // let transferTime = performance.now() - operationStart - operationTime;
        // console.log(`Read data from GPU time: ${transferTime.toFixed(2)} ms`)
        outputBuffer.set(this._operands[outputIndex].tensor.data);
      }
      resolve('execute success');
    });
  }
}

