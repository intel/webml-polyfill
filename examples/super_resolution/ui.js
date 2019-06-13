$(document).ready(() => {

  updateTitle('Super Resolution', ub, up, um, ut);

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  imageElement.addEventListener('load', () => {
    utilsPredict(imageElement, currentBackend, currentPrefer);
  }, false);
});

$(window).load(() => {

  constructModelTable(superResolutionModels);

  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})