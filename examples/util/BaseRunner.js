class BaseRunner {
  constructor() {
    this._currentModelInfo = {};
    this._currentRequest = null;
    this._progressHandler = null;
    this._bLoaded = false; // loaded status of raw model for Web NN API
    this._model = null; // get Web NN model by converting raw model
    this._bInitialized = false; // initialized status for model
    this._inferenceTime = 0.0; // unit is 'ms'
    this._labels = null; // optional for some examples
  }

  /**
   * This method is to set '_labels'.
   * @param {!Array<string>} labels An array oject that for label info.
   */
  _setLabels = (labels) => {
    this._labels = labels;
  };

  /**
   * This method is to load label file.
   * @param {string} url A string that for label file url.
   */
  _loadLabelsFile = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  /**
   * This method is to set '_currentModelInfo'.
   * @param {!Object<string, *>} modelInfo An object that for model info which was configed in modeZoo.js.
   *     An example for model info:
   *       modelInfo = {
   *         modelName: {string}, // 'MobileNet v1 (TFLite)'
   *         format: {string}, // 'TFLite'
   *         modelId: {string}, // 'mobilenet_v1_tflite'
   *         modelSize: {string}, // '16.9MB'
   *         inputSize: {!Array<number>}, // [224, 224, 3]
   *         outputSize: {number} // 1001
   *         modelFile: {string}, // '../image_classification/model/mobilenet_v1_1.0_224.tflite'
   *         labelsFile: {string}, // '../image_classification/model/labels1001.txt'
   *         preOptions: {!Obejct<string, *>}, // {mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5],}
   *         intro: {string}, // 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
   *         paperUrl: {string}, // 'https://arxiv.org/pdf/1704.04861.pdf'
   *       };
   */
  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo;
  };

  /**
   * This method is to set '_currentRequest'.
   * @param {!XMLHttpRequest} req
   */
  _setRequest = (req) => {
    // Record current request, aborts the request if it has already been sent
    this._currentRequest = req;
  };

  /**
   * This method is to set '_progressHandler'.
   * @param {function(!ProcessEvent): undefined} handler
   */
  setProgressHandler = (handler) => {
    // Use handler to prompt for model loading progress info
    this._progressHandler = handler;
  };

  /**
   * This method is to set '_bLoaded'.
   * @param {(string|boolean)} info A load model file path or a boolean that for whether model loaded .
   */
  _setLoadedFlag = (info) => {
    if (typeof info === "boolean") {
      this._bLoaded = info;
    } else {
      if (this._currentModelInfo.modelFile === info) {
        this._bLoaded = true;
      } else {
        this._bLoaded = false;
      }
    }
  };

  /**
   * This method is to set '_model'.
   * @param {!TFliteModelImporter|!OnnxModelImporter|!OpenVINOModelImporter} model
   */
  _setModel = (model) => {
    this._model = model;
  };

  /**
   * This method is to set '_bInitialized'.
   * @param {boolean} flag A boolean that for whether model initialized.
   */
  _setInitializedFlag = (flag) => {
    this._bInitialized = flag;
  };

  /**
   * This method is to set '_inferenceTime'.
   * @param {number} time
   */
  _setInferenceTime = (time) => {
    this._inferenceTime = time;
  };

  /**
   * This method is to do loading resource with specified url.
   * @param {string} url A string for url, such as model file url, label file url, etc..
   * @param {function(!ProcessEvent): undefined} handler
   * @param {boolean=} isBinary A boolean that for setting response type
   * @returns {!Promise}
   */
  _loadURL = async (url, handler = null, isBinary = false) => {
    let _this = this;
    return new Promise((resolve, reject) => {
      if (_this._currentRequest != null) {
        _this._currentRequest.abort();
      }
      let oReq = new XMLHttpRequest();
      _this._setRequest(oReq);
      oReq.open('GET', url, true);
      if (isBinary) {
        oReq.responseType = 'arraybuffer';
      }
      oReq.onload = function (ev) {
        _this._setRequest(null);
        if (oReq.readyState === 4) {
          if (oReq.status === 200) {
            resolve(oReq.response);
          } else {
            reject(new Error(`Failed to load ${url} . Status: [${oReq.status}]`));
          }
        }
      };
      if (handler != null) {
        oReq.onprogress = handler;
      }
      oReq.send();
    });
  };

  /**
   * This method is to do initialization when single instance runner loading model.
   * @param {!Object<string, *>} modelInfo An object for model info which was configed in modeZoo.js.
   *     See modelInfo details from above '_setModelInfo' method.
   */
  doInitialization = (modelInfo) => {};

  /**
   * This method is to load model file and relevant resources, such as label file.
   * @param {!Object<string, *>} modelInfo An object for model info which was configed in modeZoo.js.
   *     See modelInfo details from above '_setModelInfo' method.
   */
  loadModel = async (modelInfo, workload) => {
    if (this._bLoaded) {
      console.log(`${this._currentModelInfo.modelFile} already loaded.`);
      return;
    }

    let modelPath = this._currentModelInfo.modelFile;
    // for local workload test
    if (!modelPath.toLowerCase().startsWith("https://") && !modelPath.toLowerCase().startsWith("http://")) {
      if (typeof workload !== 'undefined') {
        modelPath = "../examples/util/" + modelPath;
      }
    }
    await this._loadModelFile(modelPath);

    if (this._currentModelInfo.labelsFile != null) {
      let labelPath = this._currentModelInfo.labelsFile;
      // for local workload test
      if (!labelPath.toLowerCase().startsWith("https://") && !labelPath.toLowerCase().startsWith("http://")) {
        if (typeof workload !== 'undefined') {
          labelPath = "../examples/util/" + labelPath;
        }
      }
      await this._loadLabelsFile(labelPath);
    }
  };

  /**
   * This method is to do compiling a machine learning model.
   * Compilation model by WebNN framework needs options paratmeter.
   * Compilation model by OpenCV.js framework doesn't need options paratmeter.
   * @param {!Object<string, *>|undefined} options
   *     options = { // for WebNN
   *       backend: {string}, // 'WASM'|'WebGL'|'WebGPU'|'WebML'
   *       prefer: {string}, // 'fast'|'sustained'|'low'|'ultra_low'
   *     };
   */
  _doCompile = (options) => {};

  /**
   * This method is to check whether model compiled.
   * @param {!Object<string, *>|undefined} options
   *     options = { // for WebNN
   *       backend: {string}, // 'WASM'|'WebGL'|'WebGPU'|'WebML'
   *       prefer: {string}, // 'fast'|'sustained'|'low'|'ultra_low'
   *     };
   */
  _checkInitializedCompilation = (options) => {
    return this._bInitialized;
  }

  /**
   * This method is to compile a machine learning model.
   * Compilation model by WebNN framework needs options paratmeter.
   * Compilation model by OpenCV.js framework doesn't need options paratmeter.
   * @param {!Object<string, *>|undefined} options
   *     options = { // for WebNN
   *       backend: {string}, // 'WASM'|'WebGL'|'WebGPU'|'WebML'
   *       prefer: {string}, // 'fast'|'sustained'|'low'|'ultra_low'
   *     };
   */
  compileModel = async (options) => {
    if (this._checkInitializedCompilation(options)) {
      console.log('Model was already compiled.');
      return;
    }

    this._setInitializedFlag(false);
    await this._doCompile(options);
    this._setInitializedFlag(true);
  };

  /**
   * This method is to get input tensor for inference.
   * @param {!Object<string, *>|!TypedArray<number>} input
   *     input = {
   *       src: !HTMLElement, //An object for HTML [<img> | <video> | <audio>] element.
   *       options: { // An object to get input tensor.
   *         // inputSize was configed in modelZoo.js, inputSize = [h, w, c] or [1, size] for audio example.
   *         inputSize: {!Array<number>},
   *         // preOptions was also configed in modelZoo.js,
   *         // preOptions= {} or likes {mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5],}
   *         preOptions: {!Object<string, *>},
   *         imageChannels: {number},
   *         drawOptions: { // optional, drawOptions is used for CanvasRenderingContext2D.drawImage() method.
   *           sx: {number}, // the x-axis coordinate of the top left corner of sub-retangle of the source image
   *           sy: {number}, // the y-axis coordinate of the top left corner of sub-retangle of the source image
   *           sWidth: {number}, // the width of the sub-retangle of the source image
   *           sHeight: {number}, // the height of the sub-retangle of the source image
   *           dWidth: {number}, // the width to draw the image in the detination canvas
   *           dHeight: {number}, // the height to draw the image in the detination canvas
   *         },
   *         scaledFlag: {boolean}, // optional, need scaled the width and height of element to get need inputTensor
   *       },
   *     };
   *     or input, likes:
   *     input = { // for Speech Command example
   *       src: {!HTMLElement}, // audio element
   *       options: {
   *         inputSize: {!Array<number>},
   *         sampleRate: {number},
   *         mfccsOptions: {!Object<string, *>}, // see details of mfccsOptions from speechCommandModels configurations of modelZoo.js
   *       },
   *     };
   *     or input is a Typed Array object, likes in Speech Recognition example.
   */
  _getInputTensor = (input) => {};

  /**
   * This methode is to do inference with input tensor and output tensor will be updated after successfully inferenced.
   */
  _doInference = () => {};

  /**
   * This method is to do inference with an HTML [<img> | <video> | <audio>] element input.
   * The id of <img> element is fixed as 'feedElement' in index.html.
   * The id of <video> element or <audio> element is fixed as 'feedMediaElement' in index.html.
   * @param {!Object<string, *>|!TypedArray<number>} input
   *     See input details from above '_getInputTensor' method.
   */
  run = async (input) => {
    await this._getInputTensor(input);
    const start = performance.now();
    await this._doInference();
    const delta = performance.now() - start;
    this._setInferenceTime(delta);
    console.log(`Compute Time: [${delta} ms]`);
  };

  /**
   * This method is to get output tensor for post processing.
   * @returns {!TypedArray<number>|!Object<string, *>} This returns a {!TypedArray<number>} object
   *     or an {!Object<string, *> object with multi tensors likes:
   *     {
   *       outputBoxTensor: {!TypedArray<number>},
   *       outputClassScoresTensor: {!TypedArray<number>},
   *     };
   */
  _getOutputTensor = () => {};

  /**
   * This method is to get inference result including inference time, computed output tensor
   * and relevant info, such as inference time, labels info, output Tensor
   * for post processing by example side after calling 'run' method.
   * @returns {!Object<string, *>} This returns an object for inference result and relevant info, likes:
   *     {
   *       inferenceTime: {number},
   *       tensor: {!TypedArray<number>|!Object<string, *>},
   *       labels: {!Array<string>},  // optional, if example needs labels info
   *     };
   */
  getOutput = () => {
    let output = {
      inferenceTime: this._inferenceTime,
      tensor: this._getOutputTensor(),
    };

    if (this._labels != null) {
      output.labels = this._labels;
    }

    return output;
  };
}
