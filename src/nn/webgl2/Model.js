import {OperationCodeToLayersMap, OperationCodeAttrsMap, WebGL2SpecialLayers} from './utils/modelUtils'
import Tensor from './Tensor'
import ops from 'ndarray-ops'
import webgl2 from './WebGL2'

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
    this.supportFeatureMapConcate = false;
  }

  /**
   * Called in nn/Compilation.js
   *
   */
  prepareModel() {
    return new Promise((resolve) => {
      if (this.supportInputLayer) {
        let attrs = {inputsNum: this._model._inputs.length};
        this._layers.push(new WebGL2SpecialLayers.Input(attrs));
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
      if (this.supportFeatureMapConcate) {
        this._layers.push(new WebGL2SpecialLayers.FeatureMapConcate({}));
      }
      // console.log(this._layers);
      resolve('compile success');
    });
  }

  /**
   * Called in nn/Execution.js
   *
   * @param {Map} inputs - input map with value: inputBuffers and indexes identifying the input operands.
   * @param {Map} outputs - output map with value: outputBuffers and indexes identifying the output operands.
   */
  execute(inputs, outputs) {
    let inputShape = this._model._operands[inputs.get(0).index].dimensions;
    let outputShape = this._model._operands[outputs.get(0).index].dimensions;
    if (inputShape.length === 4 && inputShape[0] != 1) {
      let inputIndex = 0;
      let outputIndex = 0;
      let inputSize = 0;
      let outputSize = 0;
      let inputBuffer = [];
      let outputBuffer = [];
      let tmpInputs = new Map();
      let tmpOutputs = new Map();
      let tmpBuffer = [];
      let inputDim = [];
      for (let i = 0; i< outputs.size; ++i) {
        tmpBuffer[i] = [];
      }
      for (let i = 0; i < inputShape[0]; ++i) {
        for (let j = 0; j < inputs.size; ++j) {
          inputDim = this._model._operands[inputs.get(j).index].dimensions;
          inputSize = inputDim.slice(1).reduce((accumulator, currentValue) => accumulator * currentValue);
          inputIndex = inputs.get(j).index;
          inputBuffer = inputs.get(j).buffer;
          tmpInputs.set(j, {index: inputIndex, buffer: inputBuffer.slice(inputSize * i, inputSize * (i + 1))});
        }
        for (let j = 0; j < outputs.size; ++j) {
          outputShape = this._model._operands[outputs.get(j).index].dimensions;
          outputSize = outputShape.slice(1).reduce((accumulator, currentValue) => accumulator * currentValue);
          outputIndex = outputs.get(j).index;
          outputBuffer = outputs.get(j).buffer;
          tmpOutputs.set(j, {index: outputIndex, buffer: outputBuffer.slice(outputSize * i, outputSize * (i + 1))});
        }
        this._execute(tmpInputs, tmpOutputs, i);
        for (let j = 0; j < outputs.size; ++j) {
          tmpOutputs.get(j).buffer.forEach(a => tmpBuffer[j].push(a));
        }
      }
      for (let j = 0; j < outputs.size; ++j) {
        outputs.get(j).buffer.set(tmpBuffer[j]);
      }
    } else {
      this._execute(inputs, outputs);
    }
  }

/**
   * Called in webgl2/Model.js
   *
   * @param {Map} inputs - input map with value: inputBuffers and indexes identifying the input operands.
   * @param {Map} outputs - output map with value: outputBuffers and indexes identifying the output operands.
   * @param {number} num - The number of batch.
   */
  _execute(inputs, outputs, num = 0) {
    return new Promise((resolve) => {
      let isLast = false;
      let nnOperands = this._model._operands;
      let inputBuffer = inputs.get(0).buffer;
      let inputIndex = inputs.get(0).index;
      let outputBuffer = outputs.get(0).buffer;
      let outputIndex = outputs.get(0).index;
      // let operationStart = performance.now();
      this._layers.forEach((layer, i) => {
        // let start = performance.now();
        if (i === 0) {
          let shape = nnOperands[inputIndex].dimensions;
          if (shape.length === 4) {
            shape = shape.slice(1, 4);
          } else {
            shape = shape;
          }
          if (this.supportInputLayer) {
            let inputTensors = layer.call(inputs, shape, Float32Array); 
            inputs.forEach((input, k) => {
              this._operands[input.index] = inputTensors[k];
            });
          } else {
            let inputTensor = new Tensor(inputBuffer, shape);
            this._operands[layer.outputs[0]] = layer.call(inputTensor);
          }
        } else if (i === this._layers.length - 1 && (this.supportTopClasses || this.supportFeatureMapConcate)){
          if (this.supportTopClasses) {
            let outBufferAndIndex = layer.call(this._operands[outputIndex]);
            // console.log(`outBufferAndIndex: ${outBufferAndIndex}`);
            outputBuffer.fill(0);
            let bufferLength = outBufferAndIndex.length / 2;
            for (let k = 0; k < bufferLength; ++k) {
              outputBuffer[outBufferAndIndex[k + bufferLength]] = outBufferAndIndex[k];
            }
            // console.log(`outputBuffer: ${outputBuffer}`);
          } else if (this.supportFeatureMapConcate) {
            let inputList = [];
            for (let i = 0; i < outputs.size; ++i) {
              outputIndex = outputs.get(i).index;
              inputList.push(this._operands[outputIndex]);
            }
            layer.call(inputList, outputs);
          }
        } else {
          if (layer.inputs.length === 1) {
            this._operands[layer.outputs[0]] = layer.call(this._operands[layer.inputs[0]]);
            isLast = true;
          } else {
            let MutiInputs = [];
            layer.inputs.forEach(input => {
              if (!(this._operands[input] instanceof Tensor)){
                let inputShape = nnOperands[input].dimensions;
                if (inputShape.length === 4) {
                  inputShape = inputShape.slice(1, 4);
                } else {
                  inputShape = inputShape;
                }
                let inputSize = inputShape.reduce((accumulator, currentValue) => accumulator * currentValue);
                this._operands[input] = new Tensor(nnOperands[input].value.slice(inputSize * num, inputSize * (num + 1)), inputShape);
                if (typeof(nnOperands[input].value[inputSize * (num + 1)]) === "undefined") {
                  isLast = true;
                }
              }
              if (!this._operands[input].texture && !this._operands[input].textureSlices) {
                if (this._operands[input].tensor.shape.length <= 2) {
                  this._operands[input].createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
                } else if (this._operands[input].tensor.shape.length > 2) {
                  this._operands[input].reshapeTo2D();
                  this._operands[input].createGLTexture({ type: '2d', format: 'float', supportSliceTexture: true });
                }
              }
              MutiInputs.push(this._operands[input]);
            });
            this._operands[layer.outputs[0]] = layer.call(MutiInputs);
          }
          // this._operands[layer.outputs[0]].transferFromGLTexture();
        }
        // console.log(i, (performance.now() - start).toFixed(2), layer);
      });
      if (!this.supportTopClasses && !this.supportFeatureMapConcate) {
        // let transferTime = performance.now() - operationStart - operationTime;
        for (let i = 0; i < outputs.size; ++i) {
          outputBuffer = outputs.get(i).buffer;
          outputIndex = outputs.get(i).index;
          this._operands[outputIndex].transferFromGLTexture();
          outputBuffer.set(this._operands[outputIndex].tensor.data);
        }
        // console.log(`Read data from GPU time: ${transferTime.toFixed(2)} ms`)
      }
      if (!isLast) {
        this._operands = [];
      }
      // let operationTime = performance.now() - operationStart;
      // console.log(`WebGL2 execute time: ${operationTime.toFixed(2)} ms`);
      resolve('execute success');
    });
  }

  _deleteAll() {
    webgl2.deleteAll();
    this._model._operands.forEach(operand => {
      operand.value = null;
    });
  }
}

