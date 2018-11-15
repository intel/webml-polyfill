class Utils {
  constructor() {
    this.tfModel;
    this.labels;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.progressCallback;
    this.modelFile;
    this.labelsFile;
    this.inputSize;
    this.outputSize;
    this.preOptions;
    this.postOptions;

    this.initialized = false;
  }

  async init(backend) {
    this.initialized = false;
    let result;
    if (!this.tfModel) {
      result = await this.loadModelAndLabels(this.modelFile, this.labelsFile);
      this.labels = result.text.split('\n');
      console.log(`labels: ${this.labels}`);
      let flatBuffer = new flatbuffers.ByteBuffer(result.bytes);
      this.tfModel = tflite.Model.getRootAsModel(flatBuffer);
      printTfLiteModel(this.tfModel);
    }
    this.model = new DeepLabImporter(this.tfModel, backend);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }


  async predict(imageSource) {
    if (!this.initialized) return;
    let adjustedImageShape = this.prepareInputTensor(this.inputTensor, imageSource);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    return {
      time: elapsed.toFixed(2),
      segMap: {
        data: this._argmax(this.outputTensor, 21),
        shape: adjustedImageShape
      }
    };
  }

  _argmax(array, nLabels) {
    let result = [];
    for (let i = 0; i < array.length; i += nLabels) {
      let chann = array.slice(i, i+nLabels);
      result.push(chann.indexOf(Math.max(...chann)));
    }
    return result;
  }

  async loadModelAndLabels(modelUrl, labelsUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    let text = await this.loadUrl(labelsUrl);
    return {bytes: bytes, text: text};
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
              reject(new Error('Failed to load ' + modelUrl + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.progressCallback !== 'undefined')
        request.onprogress = this.progressCallback;

      request.send();
    });
  }

  prepareInputTensor(tensor, image) {
    
    const height = this.inputSize[0];
    const width = this.inputSize[1];
    const channels = this.inputSize[2];
    const imageChannels = 4; // RGBA

    let canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    let imWidth = image.naturalWidth;
    let imHeight = image.naturalHeight;
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / 513, 1);
    let adjustedWidth = Math.floor(imWidth / resizeRatio);
    let adjustedHeight = Math.floor(imHeight / resizeRatio);
    let ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, adjustedWidth, adjustedHeight);

    // padding (replicate)
    // right
    for (let y = 0; y < adjustedHeight; y++) {
      let pixelData = ctx.getImageData(adjustedWidth-1, y, adjustedWidth, y+1);
      for (let x = adjustedWidth; x < width; x++)
        ctx.putImageData(pixelData, x, y);
    }
    // bottom
    for (let x = 0; x < adjustedWidth; x++) {
      let pixelData = ctx.getImageData(x, adjustedHeight-1, x+1, adjustedHeight);
      for (let y = adjustedHeight; y < height; y++)
        ctx.putImageData(pixelData, x, y);
    }
    // corner
    let pixelData = ctx.getImageData(adjustedWidth-1, adjustedHeight-1,
                                     adjustedWidth, adjustedHeight);
    for (let x = adjustedWidth; x < width; x++)
      for (let y = adjustedHeight; y < height; y++)
        ctx.putImageData(pixelData, x, y);

    let pixels = ctx.getImageData(0, 0, width, height).data;

    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = value / 127.5 - 1;
        }
      }
    }

    return [adjustedWidth, adjustedHeight];
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  loadModelParam(newModel) {
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.labelsFile = newModel.labelsFile;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.inputTensor = new Float32Array(newModel.inputSize.reduce((x,y) => x*y));
    this.outputTensor = new Float32Array(newModel.outputSize);
    this.tfModel = null;
  }
}
