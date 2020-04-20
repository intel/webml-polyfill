class SpeechRecognitionRunner extends BaseRunner {
  constructor() {
    super();
    this._result = {};
  }

  // The function refer with speech_sample https://github.com/opencv/dldt/blob/2020/inference-engine/samples/speech_sample/main.cpp#L166.
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
    let inputScaleFactor = this._scaleFactorForQuantization(arkData, 16384, arkRows * arkColumns);

    return {
      rows: arkRows,
      columns: arkColumns,
      data: arkData,
      inputScaleFactor: inputScaleFactor,
    }
  }

  _initError = (error) => {
    error.numScores = 0,
    error.numErrors = 0,
    error.threshold = 0.0001,
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
        let score = outputTensor[i*numColumns+j];
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
        if (error > frameError.threshold) {
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

  run = async (src) => {
    let status = 'ERROR';
    let totalTime = 0;
    let totalError = {};
    let frameError = {};
    let referenceTensor = new Float32Array(this._outputTensor[0]);

    let arkInput = await this._getTensorArrayByArk(src);
    let arkScore = await this._getTensorArrayByArk(this._currentModelInfo.scoreFile)
    this._initError(totalError);

    for (let i = 0; i < arkInput.rows; i ++) {
      this._inputTensor[0].set(arkInput.data.subarray(i * arkInput.columns, (i + 1) * arkInput.columns));
      let start = performance.now();
      status = await this._model.compute(this._inputTensor, this._outputTensor);
      let elapsed = performance.now() - start;
      totalTime += elapsed;
      referenceTensor.set(arkScore.data.subarray(i * arkScore.columns, (i + 1) * arkScore.columns));
      this._compareScores(this._outputTensor[0], referenceTensor, 1, arkScore.columns, frameError);
      this._updateScoreError(frameError, totalError);
    }

    this._result.errors = this._getReferenceCompareResults(totalError, arkScore.rows);
    this._result.cycles = arkScore.rows;
    this._result.time = totalTime.toFixed(2);

    this._setInferenceTime(totalTime);
    console.log(`Computed Status: [${status}]`);
    console.log(`Compute Time: [${totalTime} ms]`);
    return status;
  };

  _updateOutput = (output) => {
    output.result = this._result;
  };
}