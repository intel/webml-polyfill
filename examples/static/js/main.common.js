let currentBackend = getSearchParamsBackend();
let currentModel = getSearchParamsModel();
let currentPrefer = getSearchParamsPrefer();
let currentExample = getExampleDirectoryName();

const audioElement = document.getElementById('audio');
const recordElement = document.getElementById('recorder');
const videoElement = document.getElementById('video');
const imageElement = document.getElementById('image');
const inputElement = document.getElementById('input');
const progressBar = document.getElementById('progressBar');
const recordButton = document.getElementById('record');

let streaming = false;
let stats = new Stats();
let track;

const showAlert = (error) => {
  console.error(error);
  let div = document.createElement('div');
  // div.setAttribute('id', 'backendAlert');
  div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = `<strong>${error}</strong>`;
  div.innerHTML += `<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>`;
  let container = document.getElementById('container');
  container.insertBefore(div, container.firstElementChild);
}

const updateProgress = (ev) => {
  if (ev.lengthComputable) {
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    percentComplete = percentComplete.toFixed(0);
    progressBar.style = `width: ${percentComplete}%`;
    updateLoading(loadedSize.toFixed(1), totalSize.toFixed(1), percentComplete);
  }
}

const logConfig = () => {
  console.log(`Model: ${currentModel}, Backend: ${currentBackend}, Prefer: ${currentPrefer}`);
}

const errorHandler = (e) => {
  showAlert(e);
  showError(null, null);
}

const getSupportedOps = (backend, prefer) => {
  return getDefaultSupportedOps(backend, prefer);
};

let requiredOps = async () => {
  return utils.getRequiredOps();
}

const getOffloadOps = async (backend, preder) => {
  // update the global variable `supportedOps` defined in the base.js
  supportedOps = getSupportedOps(backend, preder);
  let requiredops = await requiredOps();
  let intersection = new Set([...supportedOps].filter(x => requiredops.has(x)));
  console.log('NN supported: ' + [...supportedOps]);
  console.log('Model required: ' + [...requiredops]);
  console.log('Ops offload: ' + [...intersection]);
  // Get intersection of supportedOps and requiredops
  hybridRow(currentBackend, currentPrefer, intersection);
}