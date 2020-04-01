$(document).ready(() => {

  updateTitle('ASR Sample', ub, up, um);
  constructModelTable(asrSampleModels);

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      arkFile = URL.createObjectURL(files[0]);
    }
    utilsPredict(arkFile);

  }, false);
});

$(window).load(() => {
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})