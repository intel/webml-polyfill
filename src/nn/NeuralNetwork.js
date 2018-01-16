import Model from './Model'
import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'

export default class NeuralNetwork {
  constructor() {
    this.OperationCode = OperationCode;
    this.OperandCode = OperandCode;
    this.PaddingCode = PaddingCode;
    this.PreferenceCode = PreferenceCode;
    this.FuseCode = FuseCode;
  }

  /**
   * Create a model object.
   * 
   * @param {string} name - The model name.
   */
  createModel(name) {
    return new Model(name);
  }
}