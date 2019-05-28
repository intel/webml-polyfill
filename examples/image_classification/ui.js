$(document).ready(() => {

  updateTitle('Image Classification', ub, up, um, ut);

  $('input:radio[name=bp]').click(() => {
    updateTitle('Image Classification', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    updateTitle('Image Classification', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    updateTitle('Image Classification', currentBackend, currentPrefer, `${um}`, `${ut}`);
    $('.offload').hide();
  });

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
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})