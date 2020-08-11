class SuperResolutionWebNNExecutor extends WebNNExecutor {
  constructor() {
    super();
  }

  /** @override */
  _postProcess = (data) => {
    const drawInput = (srcElement, height) => {
      const inCanvas = document.createElement('canvas');
      inCanvas.width = height;
      inCanvas.height = height;
      const inCtx = inCanvas.getContext('2d');
      inCtx.drawImage(srcElement, 0, 0, inCanvas.width, inCanvas.height);
      const outputCanvas = document.getElementById('showcanvas');
      outputCanvas.setAttribute("height", srcElement.height * 2 + 5);
      const ctx = outputCanvas.getContext('2d');
      ctx.drawImage(inCanvas, 0, 0, srcElement.width, srcElement.height);
    };

    const drawOutput = (outputTensor,srcElement, height, preOptions) => {
      const width = height;
      const mean = preOptions.mean;
      const offset = preOptions.std;
      const bytes = new Uint8ClampedArray(width * height * 4);
      const a = 255;
      for (let i = 0; i < height * width; ++i) {
        let j = i * 4;
        let r = outputTensor[i * 3] * mean[0] + offset[0];
        let g = outputTensor[i * 3 + 1] * mean[1] + offset[1];
        let b = outputTensor[i * 3 + 2] * mean[2] + offset[2];
        bytes[j + 0] = Math.round(r);
        bytes[j + 1] = Math.round(g);
        bytes[j + 2] = Math.round(b);
        bytes[j + 3] = Math.round(a);
      }
      const imageData = new ImageData(bytes, width, height);
      const inCanvas = document.createElement('canvas');
      let inCtx = inCanvas.getContext('2d');
      inCanvas.height = height;
      inCanvas.width = width;
      inCtx.putImageData(imageData, 0, 0);
      const outputCanvas = document.getElementById('showcanvas');
      const ctx = outputCanvas.getContext('2d');
      ctx.drawImage(inCanvas, 0, srcElement.height + 5, srcElement.width, srcElement.height);
    };

    const output = this._runner.getOutput();
    drawInput(this._currentInputElement, this._currentModelInfo.inputSize[0]);
    drawOutput(output.tensor, this._currentInputElement,
      this._currentModelInfo.outputSize[0], this._currentModelInfo.preOptions);
  };
}
