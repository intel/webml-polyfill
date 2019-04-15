/**
 * A profiler for measuring average time of a given number of *recurring events*
 * 
 * - Elapsed time of a single event is determined between a pair of calls to the
 *   `startEvent()` and `endEvent()`.
 * - Each time all events occur once is called an "epoch"
 */
export default class CyclicProfiler {
  /**
   * @param {number} size   Number of events to be measured. Memory used is
   *                        proportional to `size` and does not increase with
   *                        profiling time, which means that the profiler can
   *                        always be enabled in the background
   * 
   * @param {number} skipN  skip the first N epochs. Used for warm-up.
   */
  constructor(size, skipN = 0) {
    this.epochs = 0;
    this.size = size;
    this.skipN = skipN;
    this.timings = new Array(size).fill(0);

    // get profiling generator
    this.generator = this._profiling();
    // initialization. stop at the first yield stmt
    this.generator.next();
  }

  startEvent() {
    this.generator.next();
  }

  endEvent() {
    this.generator.next();
  }

  /**
   * Flush out profiling results
   * 
   * Note: It assumes that all events go through the same epochs for simplicity
   */
  flush() {
    const actualEpochs = this.epochs - this.skipN;
    const avgTimings = this.timings.map((x) => x / actualEpochs);
    this.timings.fill(0);
    this.epochs = 0;
    return {
      epochs: actualEpochs,
      elpased: avgTimings
    };
  }

  * _profiling() {
    for (this.epochs = 0; this.epochs < this.skipN; this.epochs++) {
      for (let i = 0; i < this.size; i++) {
        yield;    // dummy startEvent
        yield;    // dummy endEvent
      }
    }

    for (;;) {
      for (let i = 0; i < this.size; i++) {
        yield;    // startEvent continues from here
        const start = performance.now();
        yield;    // endEvent continues from here
        const end = performance.now();
        this.timings[i] += end - start;
      }
      this.epochs++;
    }
  }
}