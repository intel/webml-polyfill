$(document).ready(() => {

  updateTitle('Image Classification', ub, up, um);
  constructModelTable(speechCommandModels);

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    
    if (files.length > 0) {
      audioElement.src = URL.createObjectURL(files[0]);
    }
    utilsPredict(audioElement, currentBackend, currentPrefer);
  }, false);

  // audioElement.addEventListener('load', () => {
  //   utilsPredict(audioElement, currentBackend, currentPrefer);
  // }, false);

  recordeButton.addEventListener('click', () => {
    utilsPredictMicrophone();
  }, false);
});

$(window).load(() => {
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})