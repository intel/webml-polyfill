class ImageClassificationWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
  }

  /** @override */
  _postProcess = (data) => {
    const output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    const labelClasses = getTopClasses(output.tensor, output.labels, 3, deQuantizeParams);
    $('.labels-wrapper').show();
    $('.seg-label').hide();
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };
}