class imageClassificationExample extends baseCameraExample {
  constructor(models) {
    super(models);
  }

  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('video').toggleClass('fullscreen');
    });
  };

  _createRunner = () => {
    const runner = new imageClassificationRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  _processCustomOutput = () => {
    const output = this._runner.getOutput();
    const deQuantizeParams =  this._runner.getDeQuantizeParams();
    const labelClasses = getTopClasses(output.outputTensor, output.labels, 3, deQuantizeParams);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };
}