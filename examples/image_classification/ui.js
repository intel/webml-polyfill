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
  });

});

$(window).load(() => {
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})