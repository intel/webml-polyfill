const optionCompact = () => {
  for (s of $('#my-gui-container ul li .property-name')) {
    if(s.innerText.toLowerCase() == 'model') { s.setAttribute('title', 'Model: The larger the value, the larger the size of the layers, and more accurate the model at the cost of speed.'); }
    if(s.innerText.toLowerCase() == 'useatrousconv' || s.innerText == 'useAtrousConv') { s.innerText = 'AtrousConv'; s.setAttribute('title', 'UseAtrousConvOps'); }
    if(s.innerText.toLowerCase() == 'outputstride') { s.innerText = 'Stride'; s.setAttribute('title', 'OutputStride: The desired stride for the output decides output dimension of model. The higher the number, the faster the performance but slower the accuracy. '); }
    if(s.innerText.toLowerCase() == 'scalefactor') { s.innerText = 'Scale'; s.setAttribute('title', 'ScaleFactor: Scale down the image size before feed it through model, set this number lower to scale down the image and increase the speed when feeding through the network at the cost of accuracy.'); }
    if(s.innerText.toLowerCase() == 'scorethreshold') { s.innerText = 'Threshold'; s.setAttribute('title', 'ScoreThreshold: Score is the probability of keypoint and pose, set score threshold higher to reduce the number of poses to draw on image and visa versa.'); }
    if(s.innerText.toLowerCase() == 'nmsradius') { s.innerText = 'Radius'; s.setAttribute('title', 'NmsRadius: The minimal distance value between two poses under multiple poses situation. The smaller this value, the poses in image are more concentrated.'); }
    if(s.innerText.toLowerCase() == 'maxdetections') { s.innerText = 'Detections'; s.setAttribute('title', 'MaxDetections: The maximul number of poses to be detected in multiple poses situation.'); }
    if(s.innerText.toLowerCase() == 'showpose') { s.innerText = 'Pose'; s.setAttribute('title', 'ShowPose'); }
    if(s.innerText.toLowerCase() == 'showboundingbox') { s.innerText = 'Bounding'; s.setAttribute('title', 'ShowBoundingBox'); }
  }
}

$(document).ready(() => {

  if (us == 'camera') {
    currentTab = 'camera';
  } else {
    currentTab = 'image';
  }

  $('#my-gui-container ul li select').after('<div class=\'select__arrow\'></div>');
  $('#my-gui-container ul li input[type=checkbox]').after('<label class=\'\'></label>');
  // $('#my-gui-container ul li .slider').remove();
  $('#posenet ul li .c input[type=text]').attr('title', 'Update value by dragging mouse up/down on inputbox');
  $('#my-gui-container ul li.string').remove();

  optionCompact();

  const updateTitleSD = (backend, prefer) => {
    let currentprefertext = {
      fast: 'FAST_SINGLE_ANSWER',
      sustained: 'SUSTAINED_SPEED',
      low: 'LOW_POWER',
      none: 'None',
    }[prefer];

    let backendtext = backend;
    if(backendtext == 'WebML') {
      backendtext = 'WebNN';
    }
    if (backend !== 'WebML' && prefer !== 'none') {
      backendtext = backend + ' + WebNN';
    }
    $('#ictitle').html(`Skeleton Detection / ${backendtext} / ${currentprefertext}`);
  }
  updateTitleSD(ub, up);

  
  $('input:radio[name=bp]').click(() => {
    $('.alert').hide();
    let polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');

    if (polyfillId !== currentBackend) {
      $('.b-polyfill input').removeAttr('checked');
      $('.b-polyfill label').removeClass('checked');
      $('#' + polyfillId).attr('checked', 'checked');
      $('#l-' + polyfillId).addClass('checked');
    } else if (currentPrefer === 'none') {
      showAlert('Select at least one backend');
      return;
    } else {
      $('.b-polyfill input').removeAttr('checked');
      $('.b-polyfill label').removeClass('checked');
      polyfillId = 'WebML';
    }

    currentBackend = polyfillId;

    if(currentBackend === 'none' || currentBackend === '') {
      $('#option').hide();
    } else {
      $('#option').show();
      optionCompact();
    }

    updateTitleSD(currentBackend, currentPrefer);
    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    main(us === 'camera');
  });

  $('input:radio[name=bw]').click(() => {
    $('.alert').hide();

    let webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');

    if (webnnId !== currentPrefer) {
      $('.b-webnn input').removeAttr('checked');
      $('.b-webnn label').removeClass('checked');
      $('#' + webnnId).attr('checked', 'checked');
      $('#l-' + webnnId).addClass('checked');
    } else if (currentBackend === 'WebML') {
      showAlert('Select at least one backend');
      return;
    } else {
      $('.b-webnn input').removeAttr('checked');
      $('.b-webnn label').removeClass('checked');
      webnnId = 'none';
    }

    currentPrefer = webnnId;

    if(currentBackend === 'none' || currentBackend === '') {
      $('#option').hide();
    } else {
      $('#option').show();
      optionCompact();
    }

    updateTitleSD(currentBackend, currentPrefer);
    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    main(us === 'camera');
  });

  $('#extra').click(() => {
    componentToggle();
    let display;
    if (ud == '0') {
      display = '1';
      ud = '1';
    } else {
      display = '0';
      ud = '0';
    }

    let strsearch;
    if (currentBackend && currentPrefer) {
      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&s=${us}&d=${display}`;
    } else {
      strsearch = `?prefer=${up}&b=${ub}&s=${us}&d=${display}`;
    }
    window.history.pushState(null, null, strsearch);
  });
});

$(document).ready(() => {
  $('#img').click(() => {
    us = 'image';
    strsearch = `?prefer=${up}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if(currentBackend === 'none') {
      showError('No backend selected', 'Please select a backend to start prediction.');
      return;
    }
    currentTab = 'image';
    updateScenario(false);
  });

  $('#cam').click(() => {
    us = 'camera';
    strsearch = `?prefer=${up}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if(currentBackend === 'none') {
      showError('No backend selected', 'Please select a backend to start prediction.');
      return;
    }
    currentTab = 'camera';
    updateScenario(true);
  });

  $('#fullscreen i svg').click(() => {
    $('#canvasvideo').toggleClass('fullscreen');
    $('#my-gui-container').toggleClass('fullscreen');
  });

});

const updateLoadingSD = (loadedSize, totalSize, percentComplete) => {
  $('.loading-page .counter h1').html(`${loadedSize}/${totalSize} ${percentComplete}%`);
}

$(window).load(() => {
  if(currentBackend === 'none' || currentBackend === '') {
    showError('No backend selected', 'Please select a backend to start prediction.');
    $('#option').hide();
    return;
  } else {
    $('#option').show();
  }
  main(us === 'camera');
})