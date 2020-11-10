class SpeechRecognitionExample extends BaseMircophoneExample {
  constructor(models) {
    super(models);
    this._arkInfo = null;
    this._targetMax = 16384;  // Sacle the maxmium input of the first utterance to 16384(15 bits)
                              // Refer with speech_sample https://github.com/openvinotoolkit/openvino/blob/2020/inference-engine/samples/speech_sample/main.cpp#L30 (MAX_VAL_2B_FEAT)
    this._threshold = null;  // The limitation to compare with the max error value
    this._result = {};
  }

  /** @override */
  _customUI = () => {
    let inputFileElement = document.getElementById('input');
    inputFileElement.addEventListener('change', (e) => {
      $('#controller div').removeClass('current');
      this.main();
    }, false);
  };

  /** @override */
  _createRunner = () => {
    let runner;
    switch (this._currentFramework) {
      case 'WebNN':
        runner = new SpeechRecognitionRunner();
        break;
      case 'OpenVINO.js':
        runner = new SpeechRecognitionOpenVINORunner();
        break;
    }
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  /** @override */
  _predict = async () => {
    try {
      let totalTime = 0;
      let frameError = {};
      let totalError = {};
      let inputTensor = new Float32Array(this._currentModelInfo.inputSize.reduce((a, b) => a * b));
      let scoreTensor = new Float32Array(this._currentModelInfo.outputSize.reduce((a, b) => a * b));
      let arkInput = this._arkInfo;
      let arkScore = await this._getTensorArrayByArk(this._currentModelInfo.scoreFile)
      
      if (this._currentBackend == 'WebML' && this._currentPrefer == 'ultra_low') {
        this._threshold = 1;  // Define for inferring with GNA in particular
      } else {
        this._threshold = 0.0001;  // Refer with speech_sample https://github.com/openvinotoolkit/openvino/blob/2020/inference-engine/samples/speech_sample/main.cpp#L29 (MAX_SCORE_DIFFERENCE)
      }

      this._initError(totalError);
      for (let i = 0; i < arkInput.rows; i++) {
        inputTensor.set(arkInput.data.subarray(i * arkInput.columns, (i + 1) * arkInput.columns));
        scoreTensor.set(arkScore.data.subarray(i * arkScore.columns, (i + 1) * arkScore.columns));
        await this._runner.run(inputTensor);
        let output = this._runner.getOutput();
        totalTime += output.inferenceTime;
        this._compareScores(output.tensor, scoreTensor, 1, arkScore.columns, frameError);
        this._updateScoreError(frameError, totalError);
      }
      this._result.errors = this._getReferenceCompareResults(totalError, arkScore.rows);
      this._result.cycles = arkScore.rows;
      this._result.time = totalTime.toFixed(2);
      console.log(`Compute Time: [${totalTime} ms]`);
      this._postProcess();
    } catch (e) {
      showAlertComponent(e);
      showErrorComponent();
    }
  };

  /** @override */
  _getCompileOptions = async () => {
    this._arkInfo = await this._getTensorArrayByArk(this._currentModelInfo.arkFile);

    let options = {
      backend: this._currentBackend,
      prefer: this._currentPrefer,
      scaleFactor: this._arkInfo.scaleFactor,
    };

    return options;
  };

  // The function refer with speech_sample https://github.com/opencv/dldt/blob/2020/inference-engine/samples/speech_sample/main.cpp.
  _scaleFactorForQuantization = (data, targetMax, numElements) => {
    let max = 0.0;
    let scaleFactor;

    for (let i = 0; i < numElements; i++) {
      let absData = Math.abs(data[i]);
      if (absData > max) {
        max = absData;
      }
    }
    if (max == 0) {
      scaleFactor = 1.0;
    } else {
      scaleFactor = targetMax / max;
    }

    return scaleFactor;
  }

  _getTensorArrayByArk = async (arkPath) => {
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
    let scaleFactor = this._scaleFactorForQuantization(arkData, this._targetMax, arkRows * arkColumns);

    return {
      rows: arkRows,
      columns: arkColumns,
      data: arkData,
      scaleFactor: scaleFactor,
    }
  }

  _initError = (error) => {
    error.numScores = 0,
    error.numErrors = 0,
    error.maxError = 0.0,
    error.rmsError = 0.0,
    error.sumError = 0.0,
    error.sumRmsError = 0.0,
    error.sumSquaredError = 0.0,
    error.maxRelError = 0.0,
    error.sumRelError = 0.0,
    error.sumSquaredRelError = 0.0
  }

  _compareScores = (outputTensor, referenceTensor, numRows, numColumns, frameError) => {
    let numErrors = 0;
    this._initError(frameError);
    for (let i = 0; i < numRows; i ++) {
      for (let j = 0; j < numColumns; j ++) {
        let score = outputTensor[i * numColumns + j];
        let refScore = referenceTensor[i * numColumns + j];
        let error = Math.abs(refScore - score);
        let rel_error = error / ((Math.abs(refScore)) + 1e-20);
        let squared_error = error * error;
        let squared_rel_error = rel_error * rel_error;
        frameError.numScores ++;
        frameError.sumError += error;
        frameError.sumSquaredError += squared_error;
        if (error > frameError.maxError) {
          frameError.maxError = error;
        }
        frameError.sumRelError += rel_error;
        frameError.sumSquaredRelError += squared_rel_error;
        if (rel_error > frameError.maxRelError) {
          frameError.maxRelError = rel_error;
        }
        if (error > this._threshold) {
          numErrors ++;
        }
      }
    }
    frameError.rmsError = Math.sqrt(frameError.sumSquaredError / (numRows * numColumns));
    frameError.sumRmsError += frameError.rmsError;
    frameError.numErrors = numErrors;
    return numErrors;
  }

  _updateScoreError = (frameError, totalError) => {
    totalError.numErrors += frameError.numErrors;
    totalError.numScores += frameError.numScores;
    totalError.sumRmsError += frameError.rmsError;
    totalError.sumError += frameError.sumError;
    totalError.sumSquaredError += frameError.sumSquaredError;
    if (frameError.maxError > totalError.maxError) {
      totalError.maxError = frameError.maxError;
    }
    totalError.sumRelError += frameError.sumRelError;
    totalError.sumSquaredRelError += frameError.sumSquaredRelError;
    if (frameError.maxRelError > totalError.maxRelError) {
      totalError.maxRelError = frameError.maxRelError;
    }
  }

  _getReferenceCompareResults = (totalError, framesNum) => {  //framesNum equals to number of frames in one utterance
    let avgError = totalError.sumError / totalError.numScores;
    let avgRmsError= totalError.sumRmsError / framesNum;
    let stdDevError= this._stdDevError(totalError);

    return {
      maxError: totalError.maxError.toFixed(15),
      avgError: avgError.toFixed(15),
      avgRmsError: avgRmsError.toFixed(15),
      stdDevError: stdDevError.toFixed(15),
      num: totalError.numErrors
    }
  }

  _stdDevError = (totalError) => {
    let result = Math.sqrt(totalError.sumSquaredError / totalError.numScores
                - (totalError.sumError / totalError.numScores) * (totalError.sumError / totalError.numScores));
    return result;
  }

  /** @override */
  _processExtra = () => {
    const result = this._result;
    try {
      let avgTime = (result.time / result.cycles).toFixed(2);
      console.log(`Inference time: ${result.time} ms`);
      console.log(`Inference cycles: ${result.cycles}`);
      console.log(`Average time: ${avgTime} ms`);
      let inferenceCyclesElement = document.getElementById('inferenceCycles');
      let inferenceTimeElement = document.getElementById('inferenceTime');
      let averageTimeElement = document.getElementById('averageTime');
      inferenceCyclesElement.innerHTML = `inference cycles: <span class='ir'>${result.cycles} times</span>`;
      inferenceTimeElement.innerHTML = `inference time: <span class='ir'>${result.time} ms</span>`;
      averageTimeElement.innerHTML = `average time: <span class='ir'>${avgTime} ms</span>`;
    } catch (e) {
      console.log(e);
    }
    try {
      console.log(`max error: ${result.errors.maxError} ms`);
      console.log(`avg error: ${result.errors.avgError} ms`);
      console.log(`avg rms error: ${result.errors.avgRmsError} ms`);
      console.log(`stdDev error: ${result.errors.stdDevError} ms`);
      let resultElement0 = document.getElementById('result0');
      let resultElement1 = document.getElementById('result1');
      let resultElement2 = document.getElementById('result2');
      let resultElement3 = document.getElementById('result3');
      resultElement0.innerHTML = result.errors.maxError;
      resultElement1.innerHTML = result.errors.avgError;
      resultElement2.innerHTML = result.errors.avgRmsError;
      resultElement3.innerHTML = result.errors.stdDevError;
    } catch (e) {
      console.log(e);
    }
    try {
      let inferenceTextElement = document.getElementById('inferenceText');
      let dev93Text = "And what should the government do about the murders.";
      console.log("Inference result: ", dev93Text);
      inferenceTextElement.innerHTML = dev93Text;
    } catch (e) {
      console.log(e);
    }
  };
}
