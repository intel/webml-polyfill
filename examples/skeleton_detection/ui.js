let up = getUrlParam('prefer');
let ub = getUrlParam('b');
let us = getUrlParam('s');
let ud = getUrlParam('d');
let strsearch;

if (!location.search) {
  strsearch = `?prefer=none&b=none&s=image&d=0`;
  currentBackend = 'none';
  let path = location.href;
  location.href = path + strsearch;
}

const componentToggle = () => {
  $('#header-sticky-wrapper').slideToggle();
  $('#query').slideToggle();
  $('.nav-pills').slideToggle();
  $('.github-corner').slideToggle();
  $('footer').slideToggle();
  $('#extra span').toggle();
}

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
    $('.nav-pills li').removeClass('active');
    $('.nav-pills #cam').addClass('active');
    $('#imagetab').removeClass('active');
    $('#cameratab').addClass('active');
    currentTab = 'camera';
  } else {
    $('.nav-pills li').removeClass('active');
    $('.nav-pills #img').addClass('active');
    $('#cameratab').removeClass('active');
    $('#imagetab').addClass('active');
    $('#fps').html('');
    currentTab = 'image';
  }

  if (hasUrlParam('b')) {
    $('.backend input').removeAttr('checked');
    $('.backend label').removeClass('checked');
    $('#' + getUrlParam('b')).attr('checked', 'checked');
    $('#l-' + getUrlParam('b')).addClass('checked');
  }

  if (hasUrlParam('prefer')) {
    $('.prefer input').removeAttr('checked');
    $('.prefer label').removeClass('checked');
    $('#' + getUrlParam('prefer')).attr('checked', 'checked');
    $('#l-' + getUrlParam('prefer')).addClass('checked');

    if (ub == 'WASM' || ub == 'WebGL') {
      $('.ml').removeAttr('checked');
      $('.lml').removeClass('checked');
    }
  }

  $('#my-gui-container ul li select').after('<div class=\'select__arrow\'></div>');
  $('#my-gui-container ul li input[type=checkbox]').after('<label class=\'\'></label>');
  // $('#my-gui-container ul li .slider').remove();
  $('#posenet ul li .c input[type=text]').attr('title', 'Update value by dragging mouse up/down on inputbox');
  $('#my-gui-container ul li.string').remove();

  optionCompact();

  const updateTitle = (backend, prefer) => {
    let currentprefertext;
    if (backend == 'WASM' || backend == 'WebGL') {
      $('#ictitle').html(`Skeleton Detection / ${backend}`);
    } else if (backend == 'WebML') {
      if (getUrlParam('p') == 'fast') {
        prefer = 'FAST_SINGLE_ANSWER';
      } else if (getUrlParam('p') == 'sustained') {
        prefer = 'SUSTAINED_SPEED';
      } else if (getUrlParam('p') == 'low') {
        prefer = 'LOW_POWER';
      }
      $('#ictitle').html(`Skeleton Detection / WebNN / ${prefer}`);
    }
  }
  updateTitle(ub, up);

  $('input:radio[name=b]').click(() => {
    $('.alert').hide();
    let rid = $('input:radio[name="b"]:checked').attr('id');
    $('.backend input').removeAttr('checked');
    $('.backend label').removeClass('checked');
    $('#' + rid).attr('checked', 'checked');
    $('#l-' + rid).addClass('checked');

    if (rid == 'WASM' || rid == 'WebGL') {
      $('.ml').removeAttr('checked');
      $('.lml').removeClass('checked');
    }

    if (rid == 'WASM' || rid == 'WebGL') {
      currentBackend = rid;
      currentPrefer = 'none';
    } else if (rid == 'fast' || rid == 'sustained' || rid == 'low') {
      currentBackend = 'WebML';
      currentPrefer = rid;
    }

    if(currentBackend === 'none' || currentBackend === '') {
      $('#option').hide();
    } else {
      $('#option').show();
      optionCompact();
    }

    updateTitle(currentBackend, currentPrefer);
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
    $('.alert').hide();
    $('#fps').html('');
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #img').addClass('active');
    $('#imagetab').addClass('active');
    $('#cameratab').removeClass('active');
    us = 'image';
    strsearch = `?prefer=${up}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if(currentBackend === 'none') {
      showError('No backend selected', 'Please select a backend to start prediction.');
      return;
    }
    currentTab = 'image';
    main(false);
  });

  $('#cam').click(() => {
    $('.alert').hide();
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #cam').addClass('active');
    $('#cameratab').addClass('active');
    $('#imagetab').removeClass('active');
    us = 'camera';
    strsearch = `?prefer=${up}&b=${currentBackend}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if(currentBackend === 'none') {
      showError('No backend selected', 'Please select a backend to start prediction.');
      return;
    }
    currentTab = 'camera';
    main(true);
  });

  $('#fullscreen i svg').click(() => {
    $('#fullscreen i').toggle();
    toggleFullScreen();
    $('#canvasvideo').toggleClass('fullscreen');
    $('#overlay').toggleClass('video-overlay');
    $('#fps').toggleClass('fullscreen');
    $('#fullscreen i').toggleClass('fullscreen');
    $('#ictitle').toggleClass('fullscreen');
    $('#inference').toggleClass('fullscreen');
    $('#my-gui-container').toggleClass('fullscreen');
  });

});

const showProgress = async (text) => {
  $('#progressmodel').show();
  await $('#progressstep').html(text);
  $('.shoulddisplay').hide();
  $('.icdisplay').hide();
  $('#resulterror').hide();
}

const showResults = () => {
  $('#progressmodel').hide();
  $('.icdisplay').fadeIn();
  $('.shoulddisplay').fadeIn();
  $('#resulterror').hide();
}

const showError = (title, description) => {
  $('#progressmodel').hide();
  $('.icdisplay').hide();
  $('.shoulddisplay').hide();
  $('#resulterror').fadeIn();
  if (title && description) {
    $('.errortitle').html(title);
    $('.errordescription').html(description);
  } else {
    $('.errortitle').html('Prediction Failed');
    $('.errordescription').html('Please check error log for more details');
  }
}

const updateLoading = (loadedSize, totalSize, percentComplete) => {
  $('.loading-page .counter h1').html(`${loadedSize}/${totalSize} ${percentComplete}%`);
}

$(window).load(() => {
  if (ud != '0') {
    componentToggle();
  }
  if(currentBackend === 'none' || currentBackend === '') {
    showError('No backend selected', 'Please select a backend to start prediction.');
    $('#option').hide();
    return;
  } else {
    $('#option').show();
  }
  main(us === 'camera');
})