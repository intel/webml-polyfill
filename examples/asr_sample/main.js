let utils = new Utils();
utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist
let front = false;


const updateResult = (result) => {
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
    if (result.errors.num == 0) {
      let dev93Text = "Saatchi officials said the management re:structuring might accelerate \
      its efforts to persuade clients to use the firm as a one stop shop for business services."
      console.log("Inference result: ", dev93Text);
      inferenceTextElement.innerHTML = dev93Text;
    } else {
      let errorText = "Please check your input ark file!";
      console.log("Inference result: ", errorText);
      inferenceTextElement.innerHTML = errorText;
    }
  } catch (e) {
    console.log(e);
  }
}

const downloadOutput = () => {
  utils.downloadArkFile();
}

const startPredictMicrophone = async () => {
  // if (streaming) {
  //   try {
  //     stats.begin();
  //     let ret = await utils.predict(recordElement);
  //     updateResult(ret);
  //     stats.end();
  //   } catch (e) {
  //     errorHandler(e);
  //   }
  // }
  console.log('Stay Tuned.')
}

const utilsPredict = async () => {
  streaming = false;
  await showProgress('done', 'done', 'current');
  try {
    let ret = await utils.predict();
    await showProgress('done', 'done', 'done');
    showResults();
    updateResult(ret);
  }
  catch (e) {
    errorHandler(e);
  }
}

const utilsPredictMicrophone = async (backend, prefer) => {
  streaming = true;
  // await showProgress('done', 'done', 'current');
  try {
    let stream = await navigator.mediaDevices.getUserMedia({audio: true});
    track = stream.getTracks()[0];
    // start recording after 0.5s
    setTimeout(recordAndPredictMicrophone(stream), 500);
  }
  catch (e) {
    errorHandler(e);
  }
}

const recordAndPredictMicrophone = (stream) => {
  let audioRecorder = new MediaRecorder(stream, {audio:true});
  audioRecorder.ondataavailable = handleDataAvailable;
  audioRecorder.start();
  setTimeout(function() {
    audioRecorder.stop();
    // Stop webmic opened by navigator.getUserMedia after record
    if (track) {
      track.stop();
    }
  }, 1000);
}

const handleDataAvailable = (e) => {
  let buffer = [];
  buffer.push(e.data);
  let blob = new Blob(buffer, {type: 'audio/wav'});
  recordElement.src = window.URL.createObjectURL(blob);
  startPredictMicrophone(recordElement);
  showProgress('done', 'done', 'done');
  showResults();
  recordElement.play();
}

const predictPath = (microphone) => {
  (!microphone) ? utilsPredict(audioElement, currentBackend, currentPrefer) : utilsPredictMicrophone(currentBackend, currentPrefer);
}

const utilsInit = async (backend, prefer) => {
  // return immediately if model, backend, prefer are all unchanged
  let init = await utils.init(backend, prefer);
  if (init == 'NOT_LOADED') {
    return;
  }
}

const updateScenario = async (camera = false) => {
  streaming = false;
  logConfig();
  predictPath(camera);
}

const updateBackend = async (camera = false, force = false) => {
  if (force) {
    utils.initialized = false;
  }
  streaming = false;
  try { utils.deleteAll(); } catch (e) { }
  logConfig();
  await showProgress('done', 'current', 'pending');
  try {
    getOffloadOps(currentBackend, currentPrefer);
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
    predictPath(camera);
  }
  catch (e) {
    errorHandler(e);
  }
}

const main = async (microphone = false) => {
  streaming = false;
  try { utils.deleteAll(); } catch (e) { }
  logConfig();
  await showProgress('current', 'pending', 'pending');
  try {
    let model = getModelById(currentModel);
    await utils.loadModel(model);
    getOffloadOps(currentBackend, currentPrefer);
    await showProgress('done', 'current', 'pending');
    await utilsInit(currentBackend, currentPrefer);
    showSubGraphsSummary(utils.getSubgraphsSummary());
  } catch (e) {
    errorHandler(e);
  }
  predictPath(microphone);
}

