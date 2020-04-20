class SpeechRecognitionExample extends BaseMircophoneExample {
  constructor(models) {
    super(models);
  }

  _customUI = () => {
    // let _this = this;
    let inputFileElement = document.getElementById('input');
    inputFileElement.addEventListener('change', (e) => {
      $('#controller div').removeClass('current');
      this.main();
    }, false);
  };

  _createRunner = () => {
    const runner = new SpeechRecognitionRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  _predict = async () => {
    try {
      const arkFile = this._currentModelInfo.arkFile;
      await this._runner.run(arkFile);
      this._processOutput();
    } catch (e) {
      showAlertComponent(e);
      showErrorComponent();
    }
  }

  _processCustomOutput = () => {
    const result = this._runner.getOutput().result;

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
      let dev93Text = "Saatchi officials said the management re:structuring might accelerate \
      its efforts to persuade clients to use the firm as a one stop shop for business services."
      console.log("Inference result: ", dev93Text);
      inferenceTextElement.innerHTML = dev93Text;
    } catch (e) {
      console.log(e);
    }
  };
}
