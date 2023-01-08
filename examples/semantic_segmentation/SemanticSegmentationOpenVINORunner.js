class SemanticSegmentationOpenVINORunner extends OpenVINORunner {
    constructor() {
        super();
    }

    /** @override */
    _getOutputTensorTypedArray = () => {
        return Int32Array;
    };

    /** @override */
    _getOutputTensor = () => {
        let outputTensor = this._output;
        return outputTensor;
    };
}