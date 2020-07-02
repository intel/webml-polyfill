import { WebNNRunner } from '~/assets/js/webnn/util/WebNNRunner'

class SemanticSegmentationRunner extends WebNNRunner {
  constructor() {
    super();
  }

  /** @override */
  _getOutputTensorTypedArray = () => {
    return Int32Array;
  };
}

export { SemanticSegmentationRunner }