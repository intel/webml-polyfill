function main() {
  let utils = new Utils();
  const videoElement = document.getElementById('video');
  let streaming = false;
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

  let stats = new Stats();
  stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);

  navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
    video.srcObject = stream;
    utils.init().then(() => {
      updateBackend();
      streaming = true;
      startPredict();
    });
  }).catch((error) => {
    console.log('getUserMedia error: ' + error.name, error);
  });

  function startPredict() {
    stats.begin();
    utils.predict(videoElement).then(() => {
      stats.end();
      if (streaming) {
        setTimeout(startPredict, 0);
      }
    });
  }
}
