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
    backend.innerHTML = currentBackend;
  }

  function changeBackend(newBackend) {
    if (currentBackend === newBackend) {
      return;
    }
    utils.init(newBackend).then(() => {
      updateBackend();
    });
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

  navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
    video.srcObject = stream;
    utils.init().then(() => {
      updateBackend();
      streaming = true;
      startPredict();
    });
  }).catch((error) => {
    console.log('getUserMedia error: ' + error.name, error);
  })

  function startPredict() {
    utils.predict(videoElement).then(() => {
      if (streaming) {
        setTimeout(startPredict, 0);
      }
    });
  }
}
