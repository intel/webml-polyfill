class SemanticSegmentationRunner extends WebNNRunner {
  constructor() {
    super();
  }

  _getOutputTensorTypedArray = () => {
    return Int32Array;
  };
}