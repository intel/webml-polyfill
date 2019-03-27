$(document).ready(() => {

  if(um) {
    let umupper = um.replace('_4', 'x4').toUpperCase();
    updateTitle('Super Resolution', ub, up, um, ut);
  }

  $('input:radio[name=bp]').click(() => {
    let umupper = um.replace('_4', 'x4').toUpperCase();
    updateTitle('Super Resolution', currentBackend, currentPrefer, `${umupper}`, `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    let umupper = um.replace('_4', 'x4').toUpperCase();
    updateTitle('Super Resolution', currentBackend, currentPrefer, `${umupper}`, `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    let umupper = um.replace('_4', 'x4').toUpperCase();
    updateTitle('Super Resolution', currentBackend, currentPrefer, `${umupper}`, `${ut}`);
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