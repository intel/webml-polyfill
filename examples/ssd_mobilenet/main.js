const prefer = {
  MPS: 'sustained',
  BNNS: 'fast',
};

function main() {
    let utils = new Utils();
    const imageElement = document.getElementById('image');
    const inputElement = document.getElementById('input');
    const buttonEelement = document.getElementById('button');
    const backend = document.getElementById('backend');
    const wasm = document.getElementById('wasm');
    const webgl = document.getElementById('webgl');
    const webml = document.getElementById('webml');
    const selectPrefer = document.getElementById('selectPrefer');
    let currentBackend = '';
    let currentPrefer = '';
  
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
      if (newBackend !== "WebML") {
        selectPrefer.style.display = 'none';
      } else {
        selectPrefer.style.display = 'inline';
      }
      utils.deleteAll();
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

    function updatePrefer() {
      selectPrefer.innerHTML = currentPrefer === "sustained"? "MPS" : "BNNS";
    }

    function changePrefer(newPrefer) {
      if (currentPrefer === newPrefer) {
        return;
      }
      utils.deleteAll();
      selectPrefer.dataset.prefer = newPrefer;
      selectPrefer.innerHTML = 'Setting...';
      setTimeout(() => {
        utils.init(utils.model._backend).then(() => {
          currentPrefer = newPrefer;
          updatePrefer();
          utils.predict(imageElement);
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

    if (currentBackend === '') {
      if (nnNative) {
        currentBackend = 'WebML';
      } else {
        currentBackend = 'WASM';
      }
    }
  
     //register prefers
    if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
      $('.prefer').css("display","inline");
      let MPS = $('<button class="dropdown-item"/>')
        .text('MPS')
        .click(_ => changePrefer(prefer.MPS));
      $('.preference').append(MPS);
      let BNNS = $('<button class="dropdown-item"/>')
        .text('BNNS')
        .click(_ => changePrefer(prefer.BNNS));
      $('.preference').append(BNNS);
      if (!currentPrefer) {
        selectPrefer.dataset.prefer = "sustained";
        currentPrefer = "sustained";
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
  
    utils.init(currentBackend).then(() => {
      updateBackend();
      updatePrefer();
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
  