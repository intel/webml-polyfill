import PreparedModel from './PreparedModel'

export default class Device {
  /**
   * Create an instance of Device.
   */
  constructor() {}

  /**
   * Creates a prepared model for execution.
   * 
   * @param {Object} model - The model.
   * @returns {Object} The prepared model.
   */
  async prepareModel(model) {
    let preparedModel = new PreparedModel();
    await preparedModel.prepare(model);
    return preparedModel;
  }
}