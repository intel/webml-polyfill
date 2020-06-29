class ImageClassificationWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
  }

  /** @override */
  _postProcess = (data) => {
    const output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    const labelClasses = getTopClasses(output.tensor, output.labels, 3, deQuantizeParams);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
    });
  };
}