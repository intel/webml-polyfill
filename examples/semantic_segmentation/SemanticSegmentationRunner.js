class SemanticSegmentationRunner extends WebNNRunner {
  constructor() {
    super();
  }

  /** @override */
  _getOutputTensorTypedArray = () => {
    return Int32Array;
  };
}