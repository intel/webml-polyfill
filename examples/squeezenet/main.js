function main() {
  let utils = new Utils();
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  let currentBackend = '';

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Failed to setup ${backend} backend.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function updateBackend() {
    if (currentBackend === '') {
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }
    currentBackend = utils.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend).then(() => {
        updateBackend();
        utils.predict(imageElement);
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
      });
    }, 10);
  }
 
  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      changeBackend('WebML');
    }
  }

  if (nnPolyfill.supportWebGL2) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      changeBackend('WebGL2');
    }
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      changeBackend('WASM');
    }
  }

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  imageElement.onload = function() {
    utils.predict(imageElement);
  }

  utils.init().then(() => {
    updateBackend();
    utils.predict(imageElement);
  }).catch((e) => {
    console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
    console.log(e);
    showAlert(utils.model._backend);
    changeBackend('WASM');
  });
}
