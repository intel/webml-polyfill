function main() {
  let utils = new Utils();
  const videoElement = document.getElementById('video');
  const checkboxElement = document.getElementById('WebGL2');
  const checkboxLable = document.getElementById('checkboxLable');
  let streaming = false;

  checkboxElement.addEventListener('click', function(e) {
    streaming = false;
    if (checkboxElement.checked && nn.supportWebGL2) {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: true }).then(result => {
        utils.predict(videoElement, true).then(result => {
          streaming = true;
          startPredict();
        });
      }).catch(e => {
        console.error(e);
      })
    } else {
      utils.model = new MobileNet(utils.tfModel);
      utils.model.createCompiledModel( { useWebGL2: false }).then(result => {
        streaming = true;
        startPredict();
      }).catch(e => {
        console.error(e);
      })
    }
  });

  navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
    video.srcObject = stream;
    utils.init().then(() => {
      if (!nn.supportWebGL2) {
        checkboxElement.setAttribute('hidden', true);
        checkboxLable.innerHTML = 'Do not support WebGL2!';
        checkboxLable.setAttribute('class', 'alert alert-warning');
      }
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
