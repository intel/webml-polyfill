class Utils {
  constructor() {
    this.rawModel;
    this.model;
    this.inputTensor;
    this.outputTensor;
    this.modelFile;
    this.arkFile;
    this.scoreFile;
    this.inputSize;
    this.outputSize;
    this.postOptions;
    this.updateProgress;
    this.backend = '';
    this.prefer = '';
    this.initialized = false;
    this.loaded = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
  }

  async loadModel(model) {
    if (this.loaded && this.modelFile === model.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    // set new model params
    this.inputSize = model.inputSize;
    this.outputSize = model.outputSize;
    this.sampleRate = model.sampleRate;
    this.modelFile = model.modelFile;
    this.arkFile = model.arkFile;
    this.scoreFile = model.scoreFile;
    this.postOptions = model.postOptions || {};
    this.isQuantized = model.isQuantized || false;
    let typedArray;
    if (this.isQuantized) {
      typedArray = Uint8Array;
    } else {
      typedArray = Float32Array;
    }
    this.inputTensor = new typedArray(this.inputSize.reduce((a, b) => a * b));
    this.outputTensor = new typedArray(this.outputSize.reduce((a, b) => a * b));

    let arrayBuffer = await this.loadUrl(this.modelFile, true, true);
    let resultBytes = new Uint8Array(arrayBuffer);

    switch (this.modelFile.split('.').pop()) {
      case 'tflite':
        let flatBuffer = new flatbuffers.ByteBuffer(resultBytes);
        this.rawModel = tflite.Model.getRootAsModel(flatBuffer);
        this.rawModel._rawFormat = 'TFLITE';
        printTfLiteModel(this.rawModel);
        break;
      case 'onnx':
        let err = onnx.ModelProto.verify(resultBytes);
        if (err) {
          throw new Error(`Invalid model ${err}`);
        }
        this.rawModel = onnx.ModelProto.decode(resultBytes);
        this.rawModel._rawFormat = 'ONNX';
        printOnnxModel(this.rawModel);
        break;
      case 'bin':
        const networkFile = this.modelFile.replace(/bin$/, 'xml');
        const networkText = await this.loadUrl(networkFile, false, false);
        const weightsBuffer = resultBytes.buffer;
        this.rawModel = new OpenVINOModel(networkText, weightsBuffer);
        this.rawModel._rawFormat = 'OPENVINO';
        break;
      default:
        throw new Error('Unrecognized model format');
    }
    this.loaded = true;
    return 'SUCCESS';
  }

  async init(backend, prefer) {
    if (!this.loaded) {
      return 'NOT_LOADED';
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED';
    }
    this.initialized = false;
    this.backend = backend;
    this.prefer = prefer;
    let configs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
      softmax: this.postOptions.softmax || false,
    };
    switch (this.rawModel._rawFormat) {
      case 'TFLITE':
        this.model = new TFliteModelImporter(configs);
        break;
      case 'ONNX':
        this.model = new OnnxModelImporter(configs);
        break;
      case 'OPENVINO':
        this.model = new OpenVINOModelImporter(configs);
        break;
    }
    let result = await this.model.createCompiledModel();
    console.log(`compilation result: ${result}`);
    let start = performance.now();
    result = await this.model.compute([this.inputTensor], [this.outputTensor]);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;

    if (this.resolveGetRequiredOps) {
      this.resolveGetRequiredOps(this.model.getRequiredOps());
    }

    return 'SUCCESS';
  }

  async getRequiredOps() {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve);
    } else {
      return this.model.getRequiredOps();
    }
  }

  getSubgraphsSummary() {
    if (this.model._backend !== 'WebML' &&
        this.model &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  }

  async predict(arkPath) {
    if (!this.initialized) return;
    if (arkPath == undefined) arkPath = this.arkFile;

    let totalTime = 0;
    let arkInput = await this.loadKaldiArkFile(arkPath);
    let arkScore = await this.loadKaldiArkFile(this.scoreFile);
    
    for (let i=0; i<arkInput.rows; i++) {
      this.inputTensor.set(arkInput.data.subarray(i*arkInput.columns, (i+1)*arkInput.columns));
      let start = performance.now();
      await this.model.compute([this.inputTensor], [this.outputTensor]);
      let elapsed = performance.now() - start;
      totalTime += elapsed;
    }

    console.log("Avg time: ", (totalTime / arkInput.rows).toFixed(2), "ms");

    return {
      time: totalTime.toFixed(2),
      result: "Stay Tuned."
    };
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      if (this.outstandingRequest) {
        this.outstandingRequest.abort();
      }
      let request = new XMLHttpRequest();
      this.outstandingRequest = request;
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        this.outstandingRequest = null;
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

  async loadKaldiArkFile(arkPath) {
    let request = new Request(arkPath);
    let response = await fetch(request);
    let arkArrayBuffer = await response.arrayBuffer();
    let arkBytesArray = new Uint8Array(arkArrayBuffer);
    let EOF = arkBytesArray.findIndex((value) => (value == 4));  // find control-D (EOF)
    let rowsBuffer = new Uint8Array(arkBytesArray.subarray(EOF+1, EOF+5)).buffer;     // read buffer of rows
    let columnsBuffer = new Uint8Array(arkBytesArray.subarray(EOF+6, EOF+10)).buffer; // read buffer of columns
    let arkRows = new Int32Array(rowsBuffer)[0];        // read number of rows
    let arkColumns = new Int32Array(columnsBuffer)[0];  // read number of columns
    let dataLength = arkRows * arkColumns * 4;
    let dataBuffer = new Uint8Array(arkBytesArray.subarray(EOF+10, EOF+10+dataLength)).buffer;  // read buffer of data
    let arkData = new Float32Array(dataBuffer);  // read number of data

    return {
      rows: arkRows,
      columns: arkColumns,
      data: arkData
    }
  }

  downloadArkFile() {
    console.log("Convert output data to ark file.");
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }

}
