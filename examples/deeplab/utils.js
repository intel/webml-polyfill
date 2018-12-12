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

  async init(backend, prefer) {
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
    let kwargs = {
      rawModel: this.tfModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new TFliteModelImporter(kwargs);
    result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    this.outputTensor = new Float32Array(this.outputSize.reduce((x,y) => x*y));
    result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;
  }

  async predict(canvas) {
    if (!this.initialized) return;
    this.prepareInputTensor(this.inputTensor, canvas);
    this.outputTensor = new Float32Array(this.outputSize.reduce((x,y) => x*y));
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    return {
      time: elapsed,
      segMap: {
        data: this.outputTensor,
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

  prepareCanvas(canvas, imgSrc) {
    const height = this.inputSize[0];
    const width = this.inputSize[1];

    canvas.width = width;
    canvas.height = height;

    let imWidth = imgSrc.naturalWidth | imgSrc.videoWidth;
    let imHeight = imgSrc.naturalHeight | imgSrc.videoHeight;
    // assume deeplab_out.width == deeplab_out.height
    let resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    let scaledWidth = Math.floor(imWidth / resizeRatio);
    let scaledHeight = Math.floor(imHeight / resizeRatio);
    let ctx = canvas.getContext('2d');
    ctx.drawImage(imgSrc, 0, 0, scaledWidth, scaledHeight);

    return [scaledWidth, scaledHeight];
  }

  prepareInputTensor(tensor, canvas) {
    let start = performance.now();

    const height = canvas.height;
    const width = canvas.width;
    const channels = 3;
    const imageChannels = 4;
    
    const ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, width, height).data;

    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y*width*imageChannels + x*imageChannels + c];
          tensor[y*width*channels + x*channels + c] = value / 127.5 - 1;
        }
      }
    }
    console.log(`Prepare time: ${(performance.now() - start).toFixed(2)} ms`);
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
