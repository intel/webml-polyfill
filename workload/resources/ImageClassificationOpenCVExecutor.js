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
    const labelClasses = getTopClasses(output.tensor, output.labels, 3);
    labelClasses.forEach((c, i) => {
      console.log(`\tlabel: ${c.label}, probability: ${c.prob}%`);
    });
  };
}