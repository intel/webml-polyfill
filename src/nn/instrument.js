/**
 * A profiler for measuring average time of a given number of *recurring events*
 */
export default class CyclicProfiler {
  /**
   * @param {number} size   Number of events to be measured. Memory used is
   *                        proportional to `size` and does not increase with
   *                        profiling time. Therefore, the profiler can always
   *                        be enabled in the background.
   * 
   * @param {number} skipN  skip the first N epochs. Each time all events occur
   *                        once is called an "epoch". Used for warm-up.
   */
  constructor(size, skipN = 0) {
    this.epochs = 0;
    this.size = size;
    this.skipN = skipN;
    this.timings = new Array(size).fill(0);

    // get profiling generator
    this.profiler = this._initProfiler();
    // start profiler. stop at the first yield stmt
    this.profiler.next();
  }

  /**
   * Elapsed time of a single event is determined between a pair of calls to the
   * `startEvent()` and `endEvent()`
   */
  startEvent() {
    this.profiler.next();
  }

  endEvent() {
    this.profiler.next();
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

  * _initProfiler() {
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