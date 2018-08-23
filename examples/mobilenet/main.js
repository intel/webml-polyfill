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

  function updateBackend() {
    currentBackend = utils.model._backend;
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getActuralNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend) {
    if (currentBackend === newBackend) {
      return;
    }
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend).then(() => {
        updateBackend();
        utils.predict(imageElement);
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
    button.setAttribute('class', 'btn btn-primary');
    input.removeAttribute('disabled');
  });
}
