const fpsToggle = (showFPS) => {
  showFPS ? $('#fps').show() : $('#fps').hide(); 
}

$(document).ready(() => {

  if (us == 'camera') {
    fpsToggle(true);
  } else {
    fpsToggle(false);
  }

  updateTitle('Object Detection', ub, up, um, ut);
  
  $('input:radio[name=bp]').click(() => {
    updateTitle('Object Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    updateTitle('Object Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    updateTitle('Object Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
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

  videoElement.addEventListener('loadeddata', () => {
    startPredictCamera();
    showResults();
  }, false);
  
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
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})