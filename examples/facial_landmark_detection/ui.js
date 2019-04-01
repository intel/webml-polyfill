const fpsToggle = (showFPS) => {
  showFPS ? $('#fps').show() : $('#fps').hide(); 
}

$(document).ready(() => {

  if (us == 'camera') {
    fpsToggle(true);
  } else {
    fpsToggle(false);
  }

  updateTitle('Facial Landmark Detection', ub, up, um, ut);
  
  $('input:radio[name=bp]').click(() => {
    updateTitle('Facial Landmark Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    updateTitle('Facial Landmark Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    updateTitle('Facial Landmark Detection', currentBackend, currentPrefer, `${um}`, `${ut}`);
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

  $('#face_landmark_tflite').attr('checked', 'checked');
  $('#l-face_landmark_tflite').addClass('checked');
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