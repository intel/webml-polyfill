class StyleTransferExample extends BaseCameraExample {
  constructor(models) {
    super(models);
    this.maxWidth = 300;
    this.maxHeight = 300;
  }

    /** @override */
    _predict = async () => {
      const input = {
        src: this._currentInputElement,
        options: {
          inputSize: this._currentModelInfo.inputSize,
          preOptions: this._currentModelInfo.preOptions,
          imageChannels: 4,
          scaledFlag: true,
        },
      };
      await this._runner.run(input);
      this._postProcess();
    };

  _processExtra = (output) => {
    const drawInput = (srcElement) => {
      const inputCanvas = document.getElementById('inputCanvas');
      const resizeRatio = Math.max(Math.max(srcElement.width / this.maxWidth, srcElement.height / this.maxHeight), 1);
      const scaledWidth = Math.floor(srcElement.width / resizeRatio);
      const scaledHeight = Math.floor(srcElement.height / resizeRatio);
      inputCanvas.height = scaledHeight;
      inputCanvas.width = scaledWidth;
      const ctx = inputCanvas.getContext('2d');
      ctx.drawImage(srcElement, 0, 0, scaledWidth, scaledHeight);

    };

    const drawOutput = (outputTensor, height, width, srcElement) => {
      const mean = [1, 1, 1, 1];
      const offset = [0, 0, 0, 0];
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
      const resizeRatio = Math.max(Math.max(srcElement.width / width, srcElement.height / height), 1);
      const scaledWidth = Math.floor(srcElement.width / resizeRatio);
      const scaledHeight = Math.floor(srcElement.height / resizeRatio);
      const outCanvas = document.createElement('canvas');
      let outCtx = outCanvas.getContext('2d');
      outCanvas.width = scaledWidth;
      outCanvas.height = scaledHeight;
      outCtx.putImageData(imageData, 0, 0, 0, 0, outCanvas.width, outCanvas.height);

      const inputCanvas = document.getElementById('inputCanvas');
      const outputCanvas = document.getElementById('outputCanvas');
      outputCanvas.width = inputCanvas.width;
      outputCanvas.height = inputCanvas.height;
      const ctx = outputCanvas.getContext('2d');
      ctx.drawImage(outCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
    };

    drawInput(this._currentInputElement);
    drawOutput(output.tensor, this._currentModelInfo.outputSize[0], this._currentModelInfo.outputSize[1], this._currentInputElement);
  };
}
