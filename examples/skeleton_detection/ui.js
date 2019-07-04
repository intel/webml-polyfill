$(document).ready(() => {

  if (us == 'camera') {
    currentTab = 'camera';
  } else {
    currentTab = 'image';
  }

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

    if (currentprefertext === 'None') {
      $('#ictitle').html(`Skeleton Detection / ${backendtext}`);
    } else {
      $('#ictitle').html(`Skeleton Detection / ${backendtext} (${currentprefertext})`);
    }
  }
  updateTitleSD(ub, up);

  $('#backendswitch').click(() => {
    $('.alert').hide();
    let polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
    let webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');
    $('.b-polyfill input').removeAttr('checked');
    $('.b-polyfill label').removeClass('checked');
    $('.b-webnn input').removeAttr('checked');
    $('.b-webnn label').removeClass('checked');

    if(!isBackendSwitch()) {
      $('.backendtitle').html('Backend');
      if (polyfillId) {
        $('#' + polyfillId).attr('checked', 'checked');
        $('#l-' + polyfillId).addClass('checked');
        currentBackend = polyfillId;
        currentPrefer = 'none';
      } else if (webnnId) {
        $('#' + webnnId).attr('checked', 'checked');
        $('#l-' + webnnId).addClass('checked');
        currentBackend = 'WebML';
        currentPrefer = webnnId;
      } else {
        $('#WASM').attr('checked', 'checked');
        $('#l-WASM').addClass('checked');
        currentBackend = 'WASM';
        currentPrefer = 'none';
      }
    } else {
      $('.backendtitle').html('Backends');
      if (polyfillId && webnnId) {
        $('#' + polyfillId).attr('checked', 'checked');
        $('#l-' + polyfillId).addClass('checked');
        $('#' + webnnId).attr('checked', 'checked');
        $('#l-' + webnnId).addClass('checked');
        currentBackend = polyfillId;
        currentPrefer = webnnId;
      } else if (polyfillId) {
        $('#' + polyfillId).attr('checked', 'checked');
        $('#l-' + polyfillId).addClass('checked');
        $('#fast').attr('checked', 'checked');
        $('#l-fast').addClass('checked');
        currentBackend = polyfillId;
        currentPrefer = 'fast';
      } else if (webnnId) {
        $('#WASM').attr('checked', 'checked');
        $('#l-WASM').addClass('checked');
        $('#' + webnnId).attr('checked', 'checked');
        $('#l-' + webnnId).addClass('checked');
        currentBackend = 'WASM';
        currentPrefer = webnnId;
      } else {
        $('#WASM').attr('checked', 'checked');
        $('#l-WASM').addClass('checked');
        $('#fast').attr('checked', 'checked');
        $('#l-fast').addClass('checked');
        currentBackend = 'WASM';
        currentPrefer = 'fast';
      }
    }

    updateTitleSD(currentBackend, currentPrefer);
    updateBackendRadioUI(currentBackend, currentPrefer);
    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);
    if (um === 'none') {
      showError('No model selected', 'Please select a model to start prediction.');
      return;
    }
    main(us === 'camera');
  })
  
  $('input:radio[name=bp]').click(() => {
    $('.alert').hide();
    let polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
    if(isBackendSwitch()) {
      if (polyfillId !== currentBackend) {
        $('.b-polyfill input').removeAttr('checked');
        $('.b-polyfill label').removeClass('checked');
        $('#' + polyfillId).attr('checked', 'checked');
        $('#l-' + polyfillId).addClass('checked');
      } else if (currentPrefer === 'none') {
        showAlert('At least one backend required, please select other backends if needed.');
        return;
      } else {
        $('.b-polyfill input').removeAttr('checked');
        $('.b-polyfill label').removeClass('checked');
        polyfillId = 'WebML';
      }
      currentBackend = polyfillId;
      updateBackendRadioUI(currentBackend, currentPrefer);
    } else {
      $('.b-polyfill input').removeAttr('checked');
      $('.b-polyfill label').removeClass('checked');
      $('.b-webnn input').removeAttr('checked');
      $('.b-webnn label').removeClass('checked');
      $('#' + polyfillId).attr('checked', 'checked');
      $('#l-' + polyfillId).addClass('checked');
      currentBackend = polyfillId;
      currentPrefer = 'none';
    }

    $('#option').show();

    updateTitleSD(currentBackend, currentPrefer);
    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);
    main(us === 'camera');
  });

  $('input:radio[name=bw]').click(() => {
    $('.alert').hide();
    let webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');
    if(isBackendSwitch()) {
      if (webnnId !== currentPrefer) {
        $('.b-webnn input').removeAttr('checked');
        $('.b-webnn label').removeClass('checked');
        $('#' + webnnId).attr('checked', 'checked');
        $('#l-' + webnnId).addClass('checked');
      } else if (currentBackend === 'WebML') {
        showAlert('At least one backend required, please select other backends if needed.');
        return;
      } else {
        $('.b-webnn input').removeAttr('checked');
        $('.b-webnn label').removeClass('checked');
        webnnId = 'none';
      }
      currentPrefer = webnnId;
      updateBackendRadioUI(currentBackend, currentPrefer);
    }
    else {
      $('.b-polyfill input').removeAttr('checked');
      $('.b-polyfill label').removeClass('checked');
      $('.b-webnn input').removeAttr('checked');
      $('.b-webnn label').removeClass('checked');
      $('#' + webnnId).attr('checked', 'checked');
      $('#l-' + webnnId).addClass('checked');
      currentBackend = 'WebML';
      currentPrefer = webnnId;
    }

    if (currentPrefer !== 'none' && currentBackend === 'none') {
      currentBackend = 'WebML';
    }
    $('#option').show();

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

$(document).ready(function(){
  $('#sdmodel').change(() => {
    sdconfig.model = $('#sdmodel').find('option:selected').attr('value');
    main(currentTab === 'camera');
  });

  $('#sdstride').change(() => {
    sdconfig.outputStride = parseInt($('#sdstride').find('option:selected').attr('value'));
    main(currentTab === 'camera');
  });

  $('#scalefactor').change(() => {
    sdconfig.scaleFactor = parseFloat($('#scalefactor').find('option:selected').attr('value'));
    main(currentTab === 'camera');
  });

  $('#sdscorethreshold').change(() => {
    sdconfig.scoreThreshold = parseFloat($('#sdscorethreshold').val());
    utils._minScore = sdconfig.scoreThreshold;
    (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
  });

  $('#sdnmsradius').change(() => {
    sdconfig.multiPoseDetection.nmsRadius = parseInt($('#sdnmsradius').val());
    utils._nmsRadius = sdconfig.multiPoseDetection.nmsRadius;
    (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, true);
  });

  $('#sdmaxdetections').change(() => {
    sdconfig.multiPoseDetection.maxDetections = parseInt($('#sdmaxdetections').val());
    utils._maxDetection = sdconfig.multiPoseDetection.maxDetections;
    (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, true);
  });

  $('#sdshowpose').change(() => {
    sdconfig.showPose = $('#sdshowpose').prop('checked');
    (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
  });

  $('#sduseatrousconvops').change(() => {
    sdconfig.useAtrousConv = $('#sduseatrousconvops').prop('checked');
    main(currentTab === 'camera');
  });

  $('#sdshowboundingbox').change(() => {
    sdconfig.showBoundingBox = $('#sdshowboundingbox').prop('checked');
    (currentTab === 'camera') ? poseDetectionFrame() : drawResult(false, false);
  });
})

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