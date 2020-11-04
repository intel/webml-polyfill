class ImageClassificationOpenCVExecutor extends OpenCVExecutor {
  constructor() {
    super();
  }

  /** @override */
  _createRunner = () => {
    const runner = new ImageClassificationOpenCVRunner();
    return runner;
  };

  /** @override */
  _postProcess = (output) => {
    $('.labels-wrapper').show();
    $('.seg-label').hide();
    const labelClasses = getTopClasses(output.tensor, output.labels, 3);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
      let labelElement = document.getElementById(`label${i}`);
      let probElement = document.getElementById(`prob${i}`);
      labelElement.innerHTML = `${c.label}`;
      probElement.innerHTML = `${c.prob}%`;
    });
  };
}