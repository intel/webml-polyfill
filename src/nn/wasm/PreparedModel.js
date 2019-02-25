import { Mutex } from '../utils';
import { getPreparedModelWorker, getPreparedModelMain } from './WorkerIPC';

let worker = getPreparedModelWorker();

export default class PreparedModel {
  constructor() {
    this._prepared = false;
    this._mutex = new Mutex();
  }

  /**
   * Prepare for model execution.
   * 
   * @param {Object} model - A model object built by user.
   */
  async prepare(model) {
    await worker.getNNOpsInstance();
    await worker.prepare(model);
    this._prepared = true;
  }

  /**
   * Launches an asynchronous execution on a prepared model.
   * 
   * @param {Array} inputs - Inputs provided by user.
   * @param {Array} outputs - Outputs will receive results.
   */
  async execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }

    // inputs/outputs are of type Map which cannot be fully cloned to the worker
    const _inputs = Array.from(inputs.values());
    const _outputs = Array.from(outputs.values());

    await this._mutex.lock();
    const retOutputs = await worker.execute(_inputs, _outputs);
    this._mutex.release();

    outputs.forEach((output, index) => { 
      output.buffer.set(retOutputs[index].buffer)
    });
  }

  async _deleteAll() {
    await this._mutex.lock();
    await worker.deleteAll();
    this._mutex.release();
  }
}
