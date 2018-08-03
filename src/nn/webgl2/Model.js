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
    this.supportFeatureMapConcate = false;
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
   * @param {number[]} inputs - An array of indexes identifying the input operands.
   * @param {number[]} outputs - An array of indexes identifying the output operands.
   */
  execute(inputs, outputs) {
    let input_shape = this._model._operands[inputs.get(0).index].dimensions;
    let output_shape = this._model._operands[outputs.get(0).index].dimensions;
    if (input_shape.length === 4){
      let inputIndex = 0;
      let outputIndex = 0;
      let input_size = 0;
      let output_size = 0;
      let inputBuffer = [];
      let outputBuffer = [];
      let tmp_inputs = new Map();
      let tmp_outputs = new Map();
      let tmp_buffer=[];
      for (let i = 0; i< outputs.size; ++i){
        tmp_buffer[i] = [];
      }
      for (let i = 0; i < input_shape[0]; ++i){
        for (let j = 0; j < inputs.size; ++j){
          input_shape = this._model._operands[inputs.get(j).index].dimensions;
          input_size = input_shape.slice(1).reduce((accumulator,currentValue)=>accumulator*currentValue);
          inputIndex = inputs.get(j).index;
          inputBuffer = inputs.get(j).buffer;
          tmp_inputs.set(j, {index: inputIndex, buffer: inputBuffer.slice(input_size * i, input_size * (i + 1))});
        }
        for (let j = 0; j < outputs.size; ++j){
          output_shape = this._model._operands[outputs.get(j).index].dimensions;
          output_size = output_shape.slice(1).reduce((accumulator,currentValue)=>accumulator*currentValue);
          outputIndex = outputs.get(j).index;
          outputBuffer= outputs.get(j).buffer;
          tmp_outputs.set(j, {index: outputIndex, buffer: outputBuffer.slice(output_size * i, output_size * (i + 1))});
        }
        this._Execute(tmp_inputs, tmp_outputs, i);
        for (let j = 0; j < outputs.size; ++j){
          tmp_buffer[j].push(...(tmp_outputs.get(j).buffer));
        }
      }
      for (let j = 0; j < outputs.size; ++j){
        outputs.get(j).buffer.set(tmp_buffer[j]);
      }
    }else{
      this._Execute(inputs, outputs);
    }
  }

  _Execute(inputs, outputs, num){
    return new Promise((resolve) => {
      num = num || 0;
      let last = false;
      let nnOperands = this._model._operands;
      let nnOperations = this._model._operations
      let inputBuffer = inputs.get(0).buffer;
      let inputIndex = inputs.get(0).index;
      let outputBuffer = outputs.get(0).buffer;
      let outputIndex = outputs.get(0).index;
      // let operationStart = performance.now();
      this._layers.forEach((layer, i) => {
        // let start = performance.now();
        if (i == 0) {
          let shape = nnOperands[inputIndex].dimensions;
          if (shape.length === 4 ) {
            shape = shape.slice(1, 4);
          } else if (shape.length === 3 || shape.length === 2) {
            shape = shape;
          } 
          else {
            throw new Error(`the shape ${shape} is not supported`);
          }
          if (this.supportInputLayer) {
            this._operands[inputIndex] = layer.call(inputBuffer, shape, Float32Array);
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
            last = true;
          } else {
            let MutiInputs = [];
            layer.inputs.forEach(input => {
              if (!(this._operands[input] instanceof Tensor)){
                let input_shape = nnOperands[input].dimensions;
                if (input_shape.length === 4){
                  input_shape = input_shape.slice(1, 4);
                }else if (input_shape.length === 2 || 3) {
                  input_shape = input_shape;
                }else{
                  throw new Error(`the shape ${shape} is not supported`);
                }
                let input_size = input_shape.reduce((accumulator, currentValue) => accumulator * currentValue);
                this._operands[input]=new Tensor(nnOperands[input].value.slice(input_size * num, input_size * (num + 1)) , input_shape);
                if ( typeof(nnOperands[input].value[input_size * (num + 1)]) === "undefined"){
                  last = true;
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
      if (!last){
        this._operands=[];
      }
      // let operationTime = performance.now() - operationStart;
      // console.log(`WebGL2 execute time: ${operationTime.toFixed(2)} ms`);
      resolve('execute success');
    });
  }
}

