class SuperResolutionExample extends BaseCameraExample {
  constructor(models) {
    super(models);
  }

  _processExtra = (output) => {
    const drawInput = (srcElement, height) => {
      const inputCanvas = document.getElementById('inputCanvas');
      inputCanvas.height = height;
      inputCanvas.width = srcElement.width / srcElement.height * height;
      const ctx = inputCanvas.getContext('2d');
      ctx.drawImage(srcElement, 0, 0, inputCanvas.width, inputCanvas.height);
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
      const outCanvas = document.createElement('canvas');
      let outCtx = outCanvas.getContext('2d');
      outCanvas.height = height;
      outCanvas.width = width;
      outCtx.putImageData(imageData, 0, 0);

      const outputCanvas = document.getElementById('outputCanvas');
      outputCanvas.width = srcElement.width / srcElement.height * height;
      const ctx = outputCanvas.getContext('2d');
      outputCanvas.height = height;
      ctx.drawImage(outCanvas, 0, 0, outputCanvas.width, height);
    };

    drawInput(this._currentInputElement, this._currentModelInfo.inputSize[0]);
    drawOutput(output.outputTensor, this._currentInputElement,
      this._currentModelInfo.outputSize[0], this._currentModelInfo.preOptions);
  };
}
