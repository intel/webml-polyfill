export default class Execution {
  /**
   * Create an Execution to apply the given compilation.
   * 
   * @param {Compilation} compilation 
   */
  constructor(compilation) {}

  /**
   * Associate a user data with an input of the model of the Execution.
   * 
   * @param {number} index - The index of the input argument we are setting.
   * @param {TypedArray} buffer - The typed array containing the data.
   */
  setInput(index, buffer) {}

  /**
   * Associate a user buffer with an output of the model of the Execution.
   * 
   * @param {number} index - The index of output.
   * @param {TypedArray} buffer - The typed array to receive the output data.
   */
  setOutput(index, buffer) {}

  /**
   * Schedule evaluation of the execution.
   */
  async startCompute() {}
}