const fpsToggle = (showFPS) => {
  showFPS ? $('#fps').show() : $('#fps').hide();
}

$(document).ready(() => {

  if (us == 'camera') {
    fpsToggle(true);
  } else {
    fpsToggle(false);
  }

  let models = {
    faceRecognition: faceRecognitionModels,
    faceDetection: faceDetectionModels
  };

  updateTitle('Face Recognition', ub, up, um);
  constructModelTable(models, true);

  targetInputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      targetImageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  searchInputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      searchImageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  cameraInputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      cameraImageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  targetImageElement.addEventListener('load', () => {
    utilsPredict(targetImageElement, searchImageElement);
  }, false);

  searchImageElement.addEventListener('load', () => {
    utilsPredict(targetImageElement, searchImageElement);
  }, false);

  cameraImageElement.addEventListener('load', () => {
    utilsPredictCamera(cameraImageElement);
  }, false);

  $('#targetInput').hide();
  $('#searchInput').hide();
  $('#cameraImageInput').hide();
  $('#detecorCanvas').hide();
  $('#recognitionCanvas').hide();
  $('#video').hide();
});

$(document).ready(() => {
  $('#img').click(() => {
    fpsToggle(false);
  });

  $('#cam').click(() => {
    fpsToggle(true);
  });

  $('#fullscreen i svg').click(() => {
    $('#canvasshow').toggleClass('fullscreen');
  });
});

$(window).load(() => {
  if (um === 'none') {
    showError('No model selected', 'Please select face detection and face recognition models to start prediction.');
    return;
  }
  main(us === 'camera');
})
