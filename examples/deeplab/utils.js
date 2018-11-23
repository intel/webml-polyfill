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
    let scaledImageShape = this.prepareInputTensor(this.inputTensor, imageSource);
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    return {
      time: elapsed.toFixed(2),
      segMap: {
        data: this.outputTensor,
        scaledShape: scaledImageShape,
        outputShape: this.outputSize,
        labels: this.labels,
      },
    };
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
    let start = performance.now();
    let timestamp = [start];
    const height = this.inputSize[0];
    const width = this.inputSize[1];
    const channels = this.inputSize[2];
    const imageChannels = 4; // RGBA

    let canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    let imWidth = image.naturalWidth | image.videoWidth;
    let imHeight = image.naturalHeight | image.videoHeight;
    // assume width == height
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    let scaledWidth = Math.floor(imWidth / resizeRatio);
    let scaledHeight = Math.floor(imHeight / resizeRatio);
    let ctx = canvas.getContext('2d');
    timestamp.push(performance.now());
    ctx.drawImage(image, 0, 0, scaledWidth, scaledHeight);
    timestamp.push(performance.now());
    // // padding (replicate)
    // // right
    // let padWidth = width - scaledWidth;
    // let padHeight = height - scaledHeight;

    // if (padWidth > 0) {
    //   let rightEdge = ctx.getImageData(scaledWidth-1, 0, 1, scaledHeight);
    //   for (let x = scaledWidth; x < width; x++)
    //     ctx.putImageData(rightEdge, x, 0);
    // }

    // // bottom
    // if (padHeight > 0) {
    //   let bottomEdge = ctx.getImageData(0, scaledHeight-1, scaledWidth, 1);
    //   for (let y = scaledHeight; y < height; y++)
    //     ctx.putImageData(bottomEdge, 0, y);
    // }

    // // corner
    // if (padWidth > 0 && padHeight > 0) {
    //   let pixel = ctx.getImageData(scaledWidth-1, scaledHeight-1, 1, 1);
    //   let cornerPixel = pixel.data;
    //   let cornerPad = new Uint8ClampedArray(padWidth * padHeight * imageChannels);
    //   for (let i = 0; i < padWidth * padHeight * imageChannels; i += imageChannels) {
    //     cornerPad[i] = cornerPixel[0];
    //     cornerPad[i+1] = cornerPixel[1];
    //     cornerPad[i+2] = cornerPixel[2];
    //     cornerPad[i+3] = cornerPixel[3];
    //   }
    //   let cornerData = new ImageData(cornerPad, padWidth, padHeight);
    //   ctx.putImageData(cornerData, scaledWidth, scaledHeight);
    // }
    timestamp.push(performance.now());

    let pixels = ctx.getImageData(0, 0, width, height).data;
    timestamp.push(performance.now());
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = value / 127.5 - 1;
        }
      }
    }
    timestamp.push(performance.now());
    let name = ['', 'init', 'scaledown', 'padding', 'getData', 'norm'];
    for (let i = 1; i < timestamp.length; i++)
      console.log(` ${i == 1 ? '┌' : '├'} ${name[i]}: ${(timestamp[i] - timestamp[i-1]).toFixed(2)} ms`);
    console.log(`Prepare time: ${(timestamp[timestamp.length-1] - timestamp[0]).toFixed(2)} ms`);
    return [scaledWidth, scaledHeight];
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

  changeModelParam(newModel) {
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelFile = newModel.modelFile;
    this.labelsFile = newModel.labelsFile;
    this.preOptions = newModel.preOptions || {};
    this.postOptions = newModel.postOptions || {};
    this.numClasses = newModel.numClasses;
    this.inputTensor = new Float32Array(newModel.inputSize.reduce((x,y) => x*y));
    this.outputTensor = new Float32Array(newModel.outputSize.reduce((x,y) => x*y));
    this.tfModel = null;
  }
}
