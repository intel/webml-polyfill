import {PreferenceCode} from './Enums'

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
    let enumValue = PreferenceCode.enumValueOf(preference);
    if (!enumValue) {
      throw new Error(`Invalid preference value ${preference}`);
    }
    this._preference = enumValue;
  }

  /**
   * Indicate that we have finished modifying a compilation.
   */
  finish() {
    this._finished = true;
  }
}