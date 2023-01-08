class ImageClassificationExample extends BaseCameraExample {
  constructor(models) {
    super(models);
  }

  /** @override */
  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('video').toggleClass('fullscreen');
    });
  };

  /** @override */
  _createRunner = () => {
    let runner;
    switch (this._currentFramework) {
      case 'WebNN':
        runner = new WebNNRunner();
        break;
      case 'OpenCV.js':
        runner = new ImageClassificationOpenCVRunner();
        break;
      case 'OpenVINO.js':
        runner = new ImageClassificationOpenVINORunner();
        break;
    }
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _processExtra = (output) => {
    let labelClasses;
    switch (this._currentFramework) {
      case 'WebNN':
        const deQuantizeParams =  this._runner.getDeQuantizeParams();
        labelClasses = getTopClasses(output.tensor, output.labels, 3, deQuantizeParams);
        break;
      case 'OpenCV.js':
      case 'OpenVINO.js':
        labelClasses = getTopClasses(output.tensor, output.labels, 3);
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

  _resetExtraOutput = () => {
    $('#inferenceresult').hide();
    for (let i = 0; i < 3; i++) {
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = '';
      probElement.innerHTML = '';
    }
  };
}