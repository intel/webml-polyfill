function main() {
  let utils = new Utils();
  const videoElement = document.getElementById('video');
  navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then((stream) => {
    video.srcObject = stream;
    utils.init().then(() => {
      startPredict();
    });
  }).catch((error) => {
    console.log('getUserMedia error: ' + error.name, error);
  })

  function startPredict() {
    utils.predict(videoElement).then(() => {
      setTimeout(startPredict, 0);
    });
  }
}
