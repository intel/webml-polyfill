const getUrlParam = (key) => {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.get(key);
}

const hasUrlParam = (key) => {
  let searchParams = new URLSearchParams(location.search);
  return searchParams.has(key);
}

const isWebML = () => {
  if (navigator.ml && navigator.ml.getNeuralNetworkContext()) {
    if (!navigator.ml.isPolyfill) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

const toggleFullScreen = () => {
  let doc = window.document;
  let docEl = doc.documentElement;

  let requestFullScreen = docEl.requestFullscreen || docEl.mozRequestFullScreen || docEl.webkitRequestFullScreen || docEl.msRequestFullscreen;
  let cancelFullScreen = doc.exitFullscreen || doc.mozCancelFullScreen || doc.webkitExitFullscreen || doc.msExitFullscreen;

  if (!doc.fullscreenElement && !doc.mozFullScreenElement && !doc.webkitFullscreenElement && !doc.msFullscreenElement) {
    requestFullScreen.call(docEl);
  }
  else {
    cancelFullScreen.call(doc);
  }
}

const updateTitle = (name, backend, prefer, model, modeltype) => {
  model = model.replace(/_/g, ' ');
  let currentprefertext = {
    fast: 'FAST_SINGLE_ANSWER',
    sustained: 'SUSTAINED_SPEED',
    low: 'LOW_POWER',
    none: 'None',
  }[prefer];

  let backendtext = backend;
  if (backendtext == 'WebML') {
    backendtext = 'WebNN';
  }
  if (backend !== 'WebML' && prefer !== 'none') {
    backendtext = backend + ' + WebNN';
  }
  $('#ictitle').html(`${name} / ${backendtext} / ${currentprefertext} / ${model} (${modeltype})`);
}

$('#header').sticky({ topSpacing: 0, zIndex: '50' });

$(window).scroll(() => {
  if ($(this).scrollTop() > 10) {
    $('#header').fadeOut();
    $('.scrolltop').fadeIn();
  } else {
    $('#header').fadeIn();
    $('.scrolltop').fadeOut();
  }
});

$('.scrolltop, #logo a').click(() => {
  $('html, body').animate({
    scrollTop: 0
  }, 1000, 'easeInOutExpo');
  return false;
});

$(document).ready(() => {
  $('.nav-menu').superfish({
    animation: { opacity: 'show' },
    speed: 400
  });

  if ($('#nav-container').length) {
    var $mobile_nav = $('#nav-container').clone().prop({ id: 'mobile-nav' });
    let fabars = `<svg aria-hidden='true' data-prefix='fas' data-icon='bars' class='svg-inline--fa fa-bars fa-w-14' role='img' viewBox='0 0 448 512'><path fill='currentColor' d='M16 132h416c8.837 0 16-7.163 16-16V76c0-8.837-7.163-16-16-16H16C7.163 60 0 67.163 0 76v40c0 8.837 7.163 16 16 16zm0 160h416c8.837 0 16-7.163 16-16v-40c0-8.837-7.163-16-16-16H16c-8.837 0-16 7.163-16 16v40c0 8.837 7.163 16 16 16zm0 160h416c8.837 0 16-7.163 16-16v-40c0-8.837-7.163-16-16-16H16c-8.837 0-16 7.163-16 16v40c0 8.837 7.163 16 16 16z'></path></svg>`;
    let fatimes = `<svg aria-hidden='true' data-prefix='fas' data-icon='times' class='svg-inline--fa fa-times fa-w-11' role='img' viewBox='0 0 352 512'><path fill='currentColor' d='M242.72 256l100.07-100.07c12.28-12.28 12.28-32.19 0-44.48l-22.24-22.24c-12.28-12.28-32.19-12.28-44.48 0L176 189.28 75.93 89.21c-12.28-12.28-32.19-12.28-44.48 0L9.21 111.45c-12.28 12.28-12.28 32.19 0 44.48L109.28 256 9.21 356.07c-12.28 12.28-12.28 32.19 0 44.48l22.24 22.24c12.28 12.28 32.2 12.28 44.48 0L176 322.72l100.07 100.07c12.28 12.28 32.2 12.28 44.48 0l22.24-22.24c12.28-12.28 12.28-32.19 0-44.48L242.72 256z'></path></svg>`;
    let chevrondown = `<svg aria-hidden='true' data-prefix='fas' data-icon='chevron-down' class='svg-inline--fa fa-chevron-down fa-w-14' role='img' viewBox='0 0 448 512'><path fill='currentColor' d='M207.029 381.476L12.686 187.132c-9.373-9.373-9.373-24.569 0-33.941l22.667-22.667c9.357-9.357 24.522-9.375 33.901-.04L224 284.505l154.745-154.021c9.379-9.335 24.544-9.317 33.901.04l22.667 22.667c9.373 9.373 9.373 24.569 0 33.941L240.971 381.476c-9.373 9.372-24.569 9.372-33.942 0z'></path></svg>`;
    let chevronup = `<svg aria-hidden='true' data-prefix='fas' data-icon='chevron-up' class='svg-inline--fa fa-chevron-up fa-w-14' role='img' viewBox='0 0 448 512'><path fill='currentColor' d='M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z'></path></svg>`;

    $mobile_nav.find('> ul').attr({ 'class': '', 'id': '' });
    $('body').append($mobile_nav);
    $('body').prepend(`<button type='button' id='mobile-nav-toggle'><i class='fa bars'></i><i class='fa times' style='display:none;'></i></button>`);
    $('body').append(`<div id='mobile-body-overly'></div>`);
    $('#mobile-nav').find('.menu-has-children').prepend(`<i class='fa chevron-down'></i><i class='fa chevron-up' style='display:none;'></i>`);

    $('#mobile-nav-toggle i.bars').html(fabars);
    $('#mobile-nav-toggle i.times').html(fatimes);
    $('#mobile-nav .menu-has-children i.chevron-down').html(chevrondown);
    $('#mobile-nav .menu-has-children i.chevron-up').html(chevronup);

    $(document).on('click', '.menu-has-children i', (e) => {
      $(this).nextAll('ul').eq(0).slideToggle();
      $('.menu-has-children i').toggle();
    });

    $(document).on('click', '#mobile-nav-toggle', (e) => {
      $('body').toggleClass('mobile-nav-active');
      $('#mobile-nav-toggle i').toggle();
      $('#mobile-body-overly').toggle();
    });

    $(document).click((e) => {
      var container = $('#mobile-nav, #mobile-nav-toggle');
      if (!container.is(e.target) && container.has(e.target).length === 0) {
        if ($('body').hasClass('mobile-nav-active')) {
          $('body').removeClass('mobile-nav-active');
          $('#mobile-nav-toggle i').toggle();
          $('#mobile-body-overly').fadeOut();
        }
      }
    });
  } else if ($('#mobile-nav, #mobile-nav-toggle').length) {
    $('#mobile-nav, #mobile-nav-toggle').hide();
  }

  // Footer badge
  if (!isWebML()) {
    $('#WebML').addClass('dnone');
    $('#l-WebML').addClass('dnone');
    $('#webmlstatus').addClass('webml-status-false').html('not supported');
  } else {
    $('#WebML').removeClass('dnone');
    $('#l-WebML').removeClass('dnone');
    $('#webmlstatus').addClass('webml-status-true').html('supported');
  }
});

let up = getUrlParam('prefer');
let ub = getUrlParam('b');
let um = getUrlParam('m');
let ut = getUrlParam('t');
let us = getUrlParam('s');
let ud = getUrlParam('d');
let strsearch;
let skeletonDetectionPath = location.pathname.toLocaleLowerCase().indexOf('skeleton_detection');


if (!location.search) {
  if (skeletonDetectionPath > -1) {
    strsearch = `?prefer=none&b=none&s=image&d=0`;
    currentBackend = 'none';
    let path = location.href;
    location.href = path + strsearch;
  } else {
    strsearch = `?prefer=none&b=WASM&m=none&t=none&s=image&d=0`;
    let path = location.href;
    location.href = path + strsearch;
  }
}

const componentToggle = () => {
  $('#header-sticky-wrapper').slideToggle();
  $('#query').slideToggle();
  $('.nav-pills').slideToggle();
  $('.github-corner').slideToggle();
  $('footer').slideToggle();
  $('#extra span').toggle();
}

const disableModel = () => {
  if (`${um}` && `${ut}`) {
    let m_t = `${um}` + '_' + `${ut}`;
    $('.model input').attr('disabled', false)
    $('.model label').removeClass('cursordefault');
    $('#' + m_t).attr('disabled', true)
    $('#l-' + m_t).addClass('cursordefault');
  }
}

const checkedModelStyle = () => {
  if (`${um}` && `${ut}`) {
    $('.model input').removeAttr('checked');
    $('.model label').removeClass('checked');
    let m_t = `${um}` + '_' + `${ut}`;
    $('#' + m_t).attr('checked', 'checked');
    $('#l-' + m_t).addClass('checked');
  }
}

$(document).ready(() => {

  if (us == 'camera') {
    $('.nav-pills li').removeClass('active');
    $('.nav-pills #cam').addClass('active');
    $('#imagetab').removeClass('active');
    $('#cameratab').addClass('active');
  } else {
    $('.nav-pills li').removeClass('active');
    $('.nav-pills #img').addClass('active');
    $('#cameratab').removeClass('active');
    $('#imagetab').addClass('active');
    $('#fps').html('');
  }

  if (hasUrlParam('b')) {
    $('.backend input').removeAttr('checked');
    $('.backend label').removeClass('checked');
    $('#' + getUrlParam('b')).attr('checked', 'checked');
    $('#l-' + getUrlParam('b')).addClass('checked');
  }

  if (hasUrlParam('m') && hasUrlParam('t')) {
    checkedModelStyle();
  }

  if (hasUrlParam('prefer')) {
    $('.prefer input').removeAttr('checked');
    $('.prefer label').removeClass('checked');
    $('#' + getUrlParam('prefer')).attr('checked', 'checked');
    $('#l-' + getUrlParam('prefer')).addClass('checked');
  }
});


if (skeletonDetectionPath <= -1) {
  $(document).ready(() => {

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
      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);

      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }

      utils.backend = '';
      updateBackend(us === 'camera');
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
      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);

      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }

      utils.backend = '';
      updateBackend(us === 'camera');
    });

    $('input:radio[name=m]').click(() => {
      $('.alert').hide();
      let rid = $('input:radio[name="m"]:checked').attr('id');
      if (rid) {
        if (rid.indexOf('_onnx') > -1) {
          um = rid.replace('_onnx', '');
          ut = 'onnx';
        }
        if (rid.indexOf('_tflite') > -1) {
          um = rid.replace('_tflite', '');
          ut = 'tflite';
        }
      }
      if (currentBackend && currentPrefer) {
        strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      } else {
        strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      }
      // location.href = strsearch;
      window.history.pushState(null, null, strsearch);

      checkedModelStyle();
      disableModel();
      currentModel = `${um}_${ut}`;
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
        strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${display}`;
      } else {
        strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${display}`;
      }
      window.history.pushState(null, null, strsearch);
    });
  })
}

$('#fullscreen i svg').click(() => {
  $('#fullscreen i').toggle();
  toggleFullScreen();
  $('video').toggleClass('fullscreen');
  $('#overlay').toggleClass('video-overlay');
  $('#fps').toggleClass('fullscreen');
  $('#fullscreen i').toggleClass('fullscreen');
  $('#ictitle').toggleClass('fullscreen');
  $('#inference').toggleClass('fullscreen');
});

$(document).ready(() => {
  $('#img').click(() => {
    $('.alert').hide();
    $('#fps').html('');
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #img').addClass('active');
    $('#imagetab').addClass('active');
    $('#cameratab').removeClass('active');
  });

  $('#cam').click(() => {
    $('.alert').hide();
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #cam').addClass('active');
    $('#cameratab').addClass('active');
    $('#imagetab').removeClass('active');
  });
});

if (skeletonDetectionPath <= -1) {
  $(document).ready(() => {
    $('#img').click(() => {
      us = 'image';
      strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);
      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }
      updateScenario();
    });

    $('#cam').click(() => {
      us = 'camera';
      strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);
      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }
      updateScenario(true);
    });
  });
}

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
  $('.loading-page .counter h1').html(`${loadedSize}/${totalSize}MB ${percentComplete}%`);
}

$(window).load(() => {
  if (ud != '0') {
    componentToggle();
  }
  disableModel();
})