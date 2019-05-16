const BenchmarkClass = {
  'image_classification': ICBenchmark,
  'object_detection': ODBenchmark,
  'skeleton_detection': SDBenchmark,
  'semantic_segmentation': SSBenchmark
};

let inputElement = null;
let pickBtnElement = null;
let canvasElement = null;
let showCanvasElement = null;
let segCanvas = null;
let bkImageSrc = null;
let imageElement = document.getElementById('image');
let modelElement = document.getElementById('modelName');
let preferDivElement = document.getElementById('preferDiv');
let preferSelectElement = document.getElementById('preferSelect');

function getPreferString() {
  return preferSelectElement.options[preferSelectElement.selectedIndex].value;
}

function getSelectedOps() {
  return getPreferString() === 'none' ? new Set() : new Set(
    Array.from(
      document.querySelectorAll('input[name=supportedOp]:checked')).map(
        x => parseInt(x.value)));
}

function updateOpsSelect() {
  const backend = JSON.parse(configurations.value).backend;
  const prefer = preferSelect.value;
  if (backend !== 'WebNN' && prefer !== 'none') {
    // hybrid mode
    document.getElementById('supported-ops-select').style.visibility = 'visible';
  } else if (backend === 'WebNN' && prefer === 'none') {
    throw new Error('No backend selected');
  } else {
    // solo mode
    document.getElementById('supported-ops-select').style.visibility = 'hidden';
  }
  supportedOps = getSelectedOps();
}

async function main() {
    inputElement.setAttribute('class', 'disabled');
    pickBtnElement.setAttribute('class', 'btn btn-primary disabled');
    let logger = new Logger(document.querySelector('#log'));
    logger.group('Benchmark');
    try {
      let configuration = JSON.parse(document.querySelector('#configurations').selectedOptions[0].value);
      configuration.modelName = document.querySelector('#modelName').selectedOptions[0].text;
      configuration.modelClass = document.querySelector('#modelName').selectedOptions[0].className;
      configuration.iterations = Number(document.querySelector('#iterations').value) + 1;
      logger.group('Environment Information');
      logger.log(`${'UserAgent'.padStart(12)}: ${(navigator.userAgent) || '(N/A)'}`);
      logger.log(`${'Platform'.padStart(12)}: ${(navigator.platform || '(N/A)')}`);
      logger.groupEnd();
      logger.group('Configuration');
      Object.keys(configuration).forEach(key => {
        if (key === 'backend') {
          let selectedOpt = preferSelectElement.options[preferSelectElement.selectedIndex];
          let backendName = configuration[key];
          if (configuration[key].indexOf('WebNN') === 0) {
            backendName += ` + ${getNativeAPI(selectedOpt.value)}(${selectedOpt.text})`;
          }
          logger.log(`${key.padStart(12)}: ${backendName}`);
        } else {
          logger.log(`${key.padStart(12)}: ${configuration[key]}`);
        }
      });
      logger.groupEnd();
      logger.group('Run');
      let benchmark = new BenchmarkClass[configuration.modelClass](configuration.modelName, configuration.backend, configuration.iterations);
      benchmark.onExecuteSingle = (i => logger.log(`Iteration: ${i + 1} / ${configuration.iterations}`));
      let summary = await benchmark.runAsync();
      benchmark.finalize();
      logger.groupEnd();
      if (summary.profilingResults && summary.profilingResults.length) {
        logger.group('Profiling');
        summary.profilingResults.forEach((line) => logger.log(line));
        logger.groupEnd();
      }
      logger.group('Result');
      logger.log(`Inference Time: <em style="color:green;font-weight:bolder;">${summary.computeResults.mean.toFixed(2)}+-${summary.computeResults.std.toFixed(2)}</em> [ms]`);
      if (summary.decodeResults) {
        logger.log(`Decode Time: <em style="color:green;font-weight:bolder;">${summary.decodeResults.mean.toFixed(2)}+-${summary.decodeResults.std.toFixed(2)}</em> [ms]`);
      }
      logger.groupEnd();
    } catch (err) {
      logger.error(err);
    }
    inputElement.removeAttribute('disabled');
    pickBtnElement.setAttribute('class', 'btn btn-primary');
    logger.groupEnd();
  }

  function setImageSrc() {
    let inputFile = document.getElementById('input').files[0];
    if (inputFile !== undefined) {
      imageElement.src = URL.createObjectURL(inputFile);
    } else {
      let modelClass = modelElement.options[modelElement.selectedIndex].className;
      switch (modelClass) {
        case 'image_classification':
          imageElement.src = document.getElementById('imageClassificationImage').src;
          break;
        case 'object_detection':
          imageElement.src = document.getElementById('objectDetectionImage').src;
          break;
        case 'skeleton_detection':
          imageElement.src = document.getElementById('poseImage').src;
          break;
        case 'semantic_segmentation':
          imageElement.src = document.getElementById('segmentationImage').src;
          break;
        default:
          imageElement.src = document.getElementById('imageClassificationImage').src;
      }
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    inputElement = document.getElementById('input');
    pickBtnElement = document.getElementById('pickButton');
    canvasElement = document.getElementById('canvas');
    showCanvasElement = document.getElementById('showCanvas');
    segCanvas = document.getElementById('segCanvas');
    inputElement.addEventListener('change', (e) => {
      $('.labels-wrapper').empty();
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
        bkImageSrc = imageElement.src;
      }
    }, false);
    modelElement.addEventListener('change', (e) => {
      $('.labels-wrapper').empty();
      bkImageSrc = null;
      setImageSrc();
    }, false);
    let configurationsElement = document.getElementById('configurations');
    configurationsElement.addEventListener('change', (e) => {
      $('.labels-wrapper').empty();
      bkImageSrc = null;
      setImageSrc();
      if (JSON.parse(e.target.value).backend === 'WebNN') {
        document.querySelector('#preferSelect option[value=none]').disabled = true;
      } else {
        document.querySelector('#preferSelect option[value=none]').disabled = false;
      }
      updateOpsSelect();
    }, false);
    let selectAllOpsElement = document.getElementById('selectAllOps');
    selectAllOpsElement.addEventListener('click', () => {
      document.querySelectorAll('input[name=supportedOp]').forEach((x) => {
        x.checked = true;
      });
    }, false);
    let uncheckAllOpsElement = document.getElementById('uncheckAllOps');
    uncheckAllOpsElement.addEventListener('click', () => {
      document.querySelectorAll('input[name=supportedOp]').forEach((x) => {
        x.checked = false;
      });
    }, false);
    let eagerModeElement = document.getElementById('eagerMode');
    eagerModeElement.addEventListener('change', (e) => {
      eager = e.target.checked;
    }, false);
    let preferSelectElement = document.getElementById('preferSelect');
    preferSelectElement.addEventListener('change', () => {
      updateOpsSelect();
      setImageSrc();
    }, false);
    let polyfillConfigurations = [{
      backend: 'WASM',
      modelName: '',
      modelClass: '',
      iterations: 0
    },
    {
      backend: 'WebGL',
      modelName: '',
      modelClass: '',
      iterations: 0
    }];
    let webnnConfigurations = [{
      backend: 'WebNN',
      modelName: '',
      modelClass: '',
      iterations: 0
    }];
    let configurations = [];
    configurations = configurations.concat(polyfillConfigurations, webnnConfigurations);
    for (let configuration of configurations) {
      let option = document.createElement('option');
      option.value = JSON.stringify(configuration);
      option.textContent = `${configuration.backend}`;
      document.querySelector('#configurations').appendChild(option);
    }
    let button = document.querySelector('#runButton');
    button.setAttribute('class', 'btn btn-primary');
    button.addEventListener('click', main);
  });