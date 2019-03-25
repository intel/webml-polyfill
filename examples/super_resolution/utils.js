class Utils {
  constructor() {
    this.rawModel;
    this.model;
    this.initialized;
    this.updateProgress;
    this.modelFile;
    this.inputSize;
    this.outputSize;
    this.inputTensor;
    this.outputTensor;
    this.inCanvas = document.createElement('canvas');
    this.inCtx = this.inCanvas.getContext('2d');
    this.outCanvas = document.createElement('canvas');
    this.outCtx = this.outCanvas.getContext('2d');
  }

  async init(backend, prefer) {
    this.initialized = false;
    let result;
    let kwargs = {
      rawModel: null,
      backend: backend,
      prefer: prefer
    };

    if (!this.rawModel) {
      result = await this.loadTfLiteModel(this.modelFile);
      let flatBuffer = new flatbuffers.ByteBuffer(result);
      this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.rawModel);
    }
    kwargs.rawModel = this.rawModel;
    this.model = new TFliteModelImporter(kwargs);

    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(imageSource) {
    if (!this.initialized) return;
    this.inCtx.drawImage(imageSource, 0, 0,
                         this.inCanvas.width,
                         this.inCanvas.height);
    this.prepareInputTensor(this.inputTensor, this.inCanvas);
    let start = performance.now();
    let result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    return {time: elapsed.toFixed(2)};
  }

  async loadTfLiteModel(modelUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    return bytes;
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  // uint8 [0, 255] => float [-1, 1]
  prepareInputTensor(tensor, canvas) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];
    const channels = 3;
    const imageChannels = 4; // RGBA
    const [mean, offset] = [127.5, 1];
    if (canvas.width !== width || canvas.height !== height) {
      throw new Error(`canvas.width(${canvas.width}) is not ${width} or canvas.height(${canvas.height}) is not ${height}`);
    }
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, width, height).data;
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y * width * imageChannels + x * imageChannels + c];
          tensor[y * width * channels + x * channels + c] = value / mean - offset;
        }
      }
    }
  }

  drawInput(canvas, imageElement) {
    if (imageElement.width) {
      canvas.width = imageElement.width / imageElement.height * canvas.height;
    } else {
      canvas.width = imageElement.videoWidth / imageElement.videoHeight * canvas.height;
    }
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  }

  // float [-1, 1] =>  uint8 [0, 255]
  drawOutput(canvas, imageElement) {
    const height = this.outputSize[0];
    const width = this.outputSize[1];
    const [mean, offset] = [127.5, 1];
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      let r, g, b, a;
      r = (this.outputTensor[i * 3] + offset) * mean;
      g = (this.outputTensor[i * 3 + 1] + offset) * mean;
      b = (this.outputTensor[i * 3 + 2] + offset) * mean;
      a = 255;
      bytes[j + 0] = Math.round(r);
      bytes[j + 1] = Math.round(g);
      bytes[j + 2] = Math.round(b);
      bytes[j + 3] = Math.round(a);
    }
    const imageData = new ImageData(bytes, width, height);

    if (imageElement.width) {
      canvas.width = imageElement.width / imageElement.height * canvas.height;
    } else {
      canvas.width = imageElement.videoWidth / imageElement.videoHeight * canvas.height;
    }
    this.outCtx.putImageData(imageData, 0, 0);
    const ctx = canvas.getContext('2d'); 
    ctx.drawImage(this.outCanvas, 0, 0, canvas.width, canvas.height);
  }

  product(array) {
    return array.reduce((a, b) => a * b);
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  changeModelParam(newModel) {
    this.modelFile = newModel.modelFile;
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.inputTensor = new Float32Array(this.product(this.inputSize));
    this.outputTensor = new Float32Array(this.product(this.outputSize));
    this.rawModel = null;
    this.inCanvas.width = this.inputSize[1];
    this.inCanvas.height = this.inputSize[0];
    this.outCanvas.width = this.outputSize[1];
    this.outCanvas.height = this.outputSize[0];
  }
}
