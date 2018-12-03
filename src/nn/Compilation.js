import {PreferenceCode,ResultCode} from './Enums'
import Device from './wasm/Device'
import * as utils from './utils'
import Execution from './Execution'
import WebGLModel from './webgl/WebGLModel'

export default class Compilation {
  /**
   * Create a Compilation to compile the given model.
   * 
   * @param {Model} model - The model to be compiled.
   */
  constructor(model) {
    this._model = model;
    this._finished = false;
    this._preference = PreferenceCode.fast_single_answer;
    this._device = new Device;
    this._preparedModel = null;
    this._backend = model._backend;
  }


  /**
   * Create a executino from compilation.
   * 
   * @returns {Execution} - the execution object.
   */
  async createExecution() {
    if (!this._finished) {
      throw new Error('Compilation is not finished');
    }
    return new Execution(this);
  }

  /**
   * Sets the execution preference.
   * 
   * @param {number} preference - The execution preference, e.g. PreferenceCode.LOW_POWER.
   */
  setPreference(preference) {
    if (this._finished) {
      throw new Error('setPreference cant modify after compilation finished');
    }
    if (!utils.validateEnum(preference, PreferenceCode)) {
      throw new Error(`Invalid preference value ${preference}`);
    }
    this._preference = preference;
    return ResultCode.NO_ERROR;
  }

  /**
   * Indicate that we have finished modifying a compilation.
   */
  async finish() {
    switch (this._backend) {
      case 'WASM': {
        this._preparedModel = await this._device.prepareModel(this._model);
      } break;
      case 'WebGL': {
        this._preparedModel = new WebGLModel(this._model);
        await this._preparedModel.prepareModel();
      } break;
      default: {
        throw new Error(`Backend ${this._backend} is not supported`);
      }
    }
    this._finished = true;
    return ResultCode.NO_ERROR;
  }
}