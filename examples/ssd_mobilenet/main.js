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
  
    function checkPreferParam() {
      if (getOS() === 'Mac OS') {
        let preferValue = getPreferParam();
        if (preferValue === 'invalid') {
          console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
          showPreferAlert();
        }
      }
    }
  
    checkPreferParam();
  
    function showAlert(backend) {
      let div = document.createElement('div');
      div.setAttribute('id', 'backendAlert');
      div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
      div.setAttribute('role', 'alert');
      div.innerHTML = `<strong>Failed to setup ${backend} backend.</strong>`;
      div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
      let container = document.getElementById('container');
      container.insertBefore(div, container.firstElementChild);
    }
  
    function showPreferAlert() {
      let div = document.createElement('div');
      div.setAttribute('id', 'preferAlert');
      div.setAttribute('class', 'alert alert-danger alert-dismissible fade show');
      div.setAttribute('role', 'alert');
      div.innerHTML = `<strong>Invalid prefer, prefer should be 'fast' or 'sustained'.</strong>`;
      div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
      let container = document.getElementById('container');
      container.insertBefore(div, container.firstElementChild);
    }
  
    function removeAlertElement() {
      let backendAlertElem =  document.getElementById('backendAlert');
      if (backendAlertElem !== null) {
        backendAlertElem.remove();
      }
      let preferAlertElem =  document.getElementById('preferAlert');
      if (preferAlertElem !== null) {
        preferAlertElem.remove();
      }
    }
  
    function updateBackend() {
      currentBackend = utils.model._backend;
      if (getUrlParams('api_info') === 'true') {
        backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
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
        }).catch((e) => {
          console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
          console.error(e);
          showAlert(utils.model._backend);
          changeBackend('WASM');
          backend.innerHTML = 'WASM';
        });
      }, 10);
    }
   
    if (nnNative) {
      webml.setAttribute('class', 'dropdown-item');
      webml.onclick = function (e) {
        removeAlertElement();
        checkPreferParam();
        changeBackend('WebML');
      }
    }
  
    if (nnPolyfill.supportWebGL2) {
      webgl.setAttribute('class', 'dropdown-item');
      webgl.onclick = function(e) {
        removeAlertElement();
        changeBackend('WebGL2');
      }
    }
  
    if (nnPolyfill.supportWasm) {
      wasm.setAttribute('class', 'dropdown-item');
      wasm.onclick = function(e) {
        removeAlertElement();
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
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(utils.model._backend);
      changeBackend('WASM');
    });
  }
  