class BaseRunner {
  constructor() {
    this._currentModelInfo = {};
    this._inputTensor = [];
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
   * @param labels: An array oject for label information.
   */
  _setLabels = (labels) => {
    this._labels = labels;
  };

  /**
   * This method is to load label file.
   * @param url: A string for label file url.
   */
  _loadLabelsFile = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  /**
   * This method is to set '_currentModelInfo'.
   * @param modelInfo: An object for model info which was configed in modeZoo.js.
   * An example for model info:
   *   modelInfo = {
   *     modelName: 'MobileNet v1 (TFLite)',
   *     format: 'TFLite',
   *     modelId: 'mobilenet_v1_tflite',
   *     modelSize: '16.9MB',
   *     inputSize: [224, 224, 3],
   *     outputSize: 1001,
   *     modelFile: '../image_classification/model/mobilenet_v1_1.0_224.tflite',
   *     labelsFile: '../image_classification/model/labels1001.txt',
   *     preOptions: {
   *       mean: [127.5, 127.5, 127.5],
   *       std: [127.5, 127.5, 127.5],
   *     },
   *     intro: 'An efficient Convolutional Neural Networks for Mobile Vision Applications.',
   *     paperUrl: 'https://arxiv.org/pdf/1704.04861.pdf'
   *   };
   */
  _setModelInfo = (modelInfo) => {
    this._currentModelInfo = modelInfo
  };

  /**
   * This method is to set '_currentRequest'.
   * @param req: An object for XMLHttpRequest instance.
   */
  _setRequest = (req) => {
    // Record current request, aborts the request if it has already been sent
    this._currentRequest = req;
  };

  /**
   * This method is to set '_progressHandler'.
   * @param handler: A function for progress handler.
   */
  setProgressHandler = (handler) => {
    // Use handler to prompt for model loading progress info
    this._progressHandler = handler;
  };

  /**
   * This method is to set '_bLoaded'.
   * @param flag: A boolean for whether model loaded.
   */
  _setLoadedFlag = (flag) => {
    this._bLoaded = flag;
  };

  /**
   * This method is to set '_model'.
   * @param model: An object for TFliteModelImporter or OnnxModelImporter or OpenVINOModelImporter instance.
   */
  _setModel = (model) => {
    this._model = model;
  };

  /**
   * This method is to set '_bInitialized'.
   * @param flag: A boolean for whether model initialized.
   */
  _setInitializedFlag = (flag) => {
    this._bInitialized = flag;
  };

  /**
   * This method is to set '_inferenceTime'.
   * @param time: A number for time.
   */
  _setInferenceTime = (time) => {
    this._inferenceTime = time;
  };

  /**
   * This method is to do loading resource with specified url.
   * @param url: A string for url, such as model file url, label file url, etc..
   * @param handler: A function for progress handler.
   * @returns {object} This returns a Promise object.
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
   * This method is to load model file and relevant resources, such as label file.
   * @param modelInfo: An object for model info.
   */
  loadModel = async (modelInfo) => {
    // Override by inherited
  };

  /**
   * This method is to compile model after calling 'loadModel' for doing inference with model.
   * @param options: {object | undefined}, detail explanations see inherited 'compileModel' method.
   */

  /**
   * This method is to compile a machine learning model.
   *   Compliation model by WebNN framework needs options paratmeter.
   *   Compilation model by OpenCV.js framework doesn't need options paratmeter.
   * @param options: {object | undefined}
   * An object for options has backend and prefer info.
   * options = {
   *   backend: backend, // 'WASM' | 'WebGL' | 'WebML'
   *   prefer: prefer, // 'fast' | 'sustained' | 'low' | 'ultra_low'
   * }
   */
  compileModel = async (options) => {
    // Override by inherited
  };

  /**
   * This method is to do inference with input src (HTML [<img> | <video> | <audio>] element).
   * The id of <img> element is fixed as 'feedElement' in index.html.
   * The id of <video> element or <audio> element is fixed as 'feedMediaElement' in index.html.
   * @param src: An object for HTML [<img> | <video> | <audio>] element.
   * @param options: An object to get input tensor. Details:
   * options = {
   *   // inputSize was configed in modelZoo.js, inputSize = [h, w, c] or [1, size] for audio example.
   *   inputSize: inputSize,
   *   // preOptions was also configed in modelZoo.js,
   *   // preOptions= {} or {mean: [number, number, number, number],std: [number, number, number, number]}
   *   preOptions: preOptions,
   *   imageChannels: 4,
   *   drawOptions: { // optional, drawOptions is used for CanvasRenderingContext2D.drawImage() method.
   *     sx: sx, // the x-axis coordinate of the top left corner of sub-retangle of the source image
   *     sy: sy, // the y-axis coordinate of the top left corner of sub-retangle of the source image
   *     sWidth: sWidth, // the width of the sub-retangle of the source image
   *     sHeight: sHeight, // the height of the sub-retangle of the source image
   *     dWidth: dWidth, // the width to draw the image in the detination canvas
   *     dHeight: dWidth, // the height to draw the image in the detination canvas
   *   },
   *   scaledFlag: true, // optional, need scaled the width and height of element to get need inputTensor
   * };
   */
  run = async (src, options) => {
    // Override by inherited
  };

  /**
   * This method is to output inference tensor for post processing by example side.
   * @param output: An object for output which will be updated with output tensor info by this method.
   */
  _getOutputTensor = (output) => {
    // Override by inherited if needed
  };

  /**
   * This method is to get inference result including inference time, computed output tensor
   * and relevant info, such as inference time, labels info, output Tensor
   * for post processing by example side after calling 'run' method.
   * @returns {object} This returns an object for inference result and relevant info.
   */
  getOutput = () => {
    let output = {
      inferenceTime: this._inferenceTime,
    };

    if (this._labels != null) {
      output.labels = this._labels;
    }

    this._getOutputTensor(output);
    return output;
  };
}
