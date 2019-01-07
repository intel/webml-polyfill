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
      if (currentOS === 'Mac OS') {
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
      div.innerHTML = `<strong>Currently ${backend} backend doesn't support SSD-MobileNet Model.</strong>`;
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
      if (getUrlParams('api_info') === 'true') {
        backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI(currentPrefer) : currentBackend;
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
        utils.init(newBackend, currentPrefer).then(() => {
          currentBackend = newBackend;
          updateBackend();
          updatePrefer();
          utils.predict(imageElement);
        }).catch((e) => {
          console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
          console.error(e);
          showAlert(utils.model._backend);
          changeBackend('WASM');
          updatePrefer();
          backend.innerHTML = 'WASM';
        });
      }, 10);
    }

    function updatePrefer() {
      selectPrefer.innerHTML = preferMap[currentPrefer];
    }

    function changePrefer(newPrefer, force) {
      if (currentPrefer === newPrefer && !force) {
        return;
      }
      utils.deleteAll();
      removeAlertElement();
      selectPrefer.innerHTML = 'Setting...';
      setTimeout(() => {
        utils.init(currentBackend, newPrefer).then(() => {
          currentPrefer = newPrefer;
          updatePrefer();
          updateBackend();
          utils.predict(imageElement);
        }).catch((e) => {
          let currentBackend = 'WebML/' + getNativeAPI(currentPrefer);
          let nextBackend = 'WebML/' + getNativeAPI(newPrefer);
          console.warn(`Failed to change backend ${nextBackend}, switch back to ${currentBackend}`);
          console.error(e);
          changePrefer(currentPrefer, true);
          showAlert(nextBackend);
          updatePrefer();
          updateBackend();
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

    if (nnPolyfill.supportWebGL) {
      webgl.setAttribute('class', 'dropdown-item');
      webgl.onclick = function(e) {
        removeAlertElement();
        changeBackend('WebGL');
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

    // register prefers
    if (currentBackend === 'WebML') {
      $('.prefer').css("display","inline");
      let sustained = $('<button class="dropdown-item"/>')
        .text('SUSTAINED_SPEED')
        .click(_ => changePrefer('sustained'));
      $('.preference').append(sustained);
      if (currentOS === 'Android') {
        let fast = $('<button class="dropdown-item"/>')
          .text('FAST_SINGLE_ANSWER')
          .click(_ => changePrefer('fast'));
        $('.preference').append(fast);
        let low = $('<button class="dropdown-item"/>')
          .text('LOW_POWER')
          .click(_ => changePrefer('low'));
        $('.preference').append(low);
      } else if (currentOS === 'Windows' || currentOS === 'Linux') {
        let fast = $('<button class="dropdown-item" disabled />')
          .text('FAST_SINGLE_ANSWER')
          .click(_ => changePrefer('fast'));
        $('.preference').append(fast);
        let low = $('<button class="dropdown-item" disabled />')
          .text('LOW_POWER')
          .click(_ => changePrefer('low'));
        $('.preference').append(low);
      } else if (currentOS === 'Mac OS') {
        let fast = $('<button class="dropdown-item"/>')
          .text('FAST_SINGLE_ANSWER')
          .click(_ => changePrefer('fast'));
        $('.preference').append(fast);
        let low = $('<button class="dropdown-item" disabled />')
          .text('LOW_POWER')
          .click(_ => changePrefer('low'));
        $('.preference').append(low);
      }
      if (!currentPrefer) {
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

    utils.init(currentBackend, currentPrefer).then(() => {
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