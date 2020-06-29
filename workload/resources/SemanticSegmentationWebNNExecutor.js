class SemanticSegmentationWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
  }

  /** @override */
  _createRunner = () => {
    const runner = new SemanticSegmentationRunner();
    return runner;
  };

  /** @override */
  _postProcess = (data) => {
    const output =  this._runner.getOutput();
    const width = this._currentModelInfo.inputSize[1];
    const imWidth = this._currentInputElement.naturalWidth | this._currentInputElement.videoWidth;
    const imHeight = this._currentInputElement.naturalHeight | this._currentInputElement.videoHeight;
    const resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    const scaledWidth = Math.floor(imWidth / resizeRatio);
    const scaledHeight = Math.floor(imHeight / resizeRatio);
    const segMap = {
      data: output.tensor,
      outputShape: this._currentModelInfo.outputSize,
      labels: output.labels,
    };
    let segCanvasElement = document.createElement('canvas');
    let renderer = new Renderer(segCanvasElement);
    renderer.setup();
    renderer.uploadNewTexture(this._currentInputElement, [scaledWidth, scaledHeight]);
    renderer.drawOutputs(segMap);
    renderer.highlightHoverLabel(this._hoverPos);
    let showCanvasElement = document.getElementById('showcanvas');
    let showCanvasContext = showCanvasElement.getContext('2d');
    showCanvasContext.drawImage(segCanvasElement, 0, 0, imWidth, imHeight);
  };
}
