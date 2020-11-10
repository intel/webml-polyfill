class SpeechRecognitionOpenVINORunner extends OpenVINORunner {
  constructor() {
    super();
  }

  _getInputTensor = (input) => {
    let infer_req = this._execNet.createInferRequest();
    const input_blob = infer_req.getBlob(this._inputInfo.name());
    const input_data = new Float32Array(input_blob.wmap());

    for(let index = 0; index < input.length; index++) {
      input_data[index] = input[index];
    }
    input_blob.unmap();
    this._inferReq = infer_req;
  };

  _getOutputTensor = () => {
    let outputTensor = this._output;
    return outputTensor;
  };
}