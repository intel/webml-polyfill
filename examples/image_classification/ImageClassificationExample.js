class ImageClassificationExample extends BaseCameraExample {
  constructor(models) {
    super(models);
  }

  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('video').toggleClass('fullscreen');
    });
  };

  _createRunner = () => {
    let runner;
    switch (this._currentFramework) {
      case 'WebNN':
        runner = new ImageClassificationWebNNRunner();
        break;
      case 'OpenCV.js':
        runner = new ImageClassificationOpenCVRunner();
        break;
    }
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  _processCustomOutput = () => {
    const output = this._runner.getOutput();
    let labelClasses;
    switch (this._currentFramework) {
      case 'WebNN':
        const deQuantizeParams =  this._runner.getDeQuantizeParams();
        labelClasses = getTopClasses(output.outputTensor, output.labels, 3, deQuantizeParams);
        break;
      case 'OpenCV.js':
        labelClasses = getTopClasses(output.outputTensor, output.labels, 3);
        break;
    }
    $('#inferenceresult').show();
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };

  _resetCustomOutput = () => {
    $('#inferenceresult').hide();
    for (let i = 0; i < 3; i++) {
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = '';
      probElement.innerHTML = '';
    }
  };
}