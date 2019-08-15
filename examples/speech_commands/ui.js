$(document).ready(() => {

  updateTitle('Speech Commands', ub, up, um);
  constructModelTable(speechCommandModels);

  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    
    if (files.length > 0) {
      audioElement.src = URL.createObjectURL(files[0]);
    }
    utilsPredict(audioElement, currentBackend, currentPrefer);
    $('#controller div').removeClass('current');

  }, false);

  $('#controller div').click(function(){
    let t = $(this).attr('id');
    audioElement.src = `audio/${t}.wav`;
    audioElement.play();
    utilsPredict(audioElement, currentBackend, currentPrefer);
    $('#controller div').removeClass('current');
    $(`#${t}`).addClass('current');
  });

  // audioElement.addEventListener('load', () => {
  //   utilsPredict(audioElement, currentBackend, currentPrefer);
  // }, false);

  recordButton.addEventListener('click', () => {
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