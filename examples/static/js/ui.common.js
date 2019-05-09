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

let up = getUrlParam('prefer');
let ub = getUrlParam('b');
let um = getUrlParam('m');
let ut = getUrlParam('t');
let us = getUrlParam('s');
let ud = getUrlParam('d');
let strsearch;
let skeletonDetectionPath = location.pathname.toLocaleLowerCase().indexOf('skeleton_detection');
let facialLandmarkDetectionPath = location.pathname.toLocaleLowerCase().indexOf('facial_landmark_detection');


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

const hybridRow = (currentBackend, currentPrefer, offloadops) => {
  if(offloadops && offloadops.size > 0 && currentBackend != 'WebML' && currentPrefer != 'none') {
    $('.offload').fadeIn();
    let offloadopsvalue = '';
    offloadops.forEach((value) => {
      let t = '<span class="ol">' + operationTypes[value] + '</span>';
      offloadopsvalue += t;
    })
    $(".ol").remove();
    $("#offloadops").html(`Following ops were offloaded to <span id='nnbackend' class='ols'></span> from <span id='polyfillbackend' class='ols'></span>: `);
    $("#offloadops").append(offloadopsvalue).append(`<span data-toggle="modal" class="subgraph-btn" data-target="#subgraphModal">View Subgraphs</span>`);
    $("#nnbackend").html(currentPrefer);
    $("#polyfillbackend").html(currentBackend);
  } else {
    $('.offload').hide();
  }
}

const showSubGraphsSummary = (summary) => {
  if(summary) {
    let listhtml = '';
    for(let i in summary) {
      let backend = summary[i].split(':')[0].toLowerCase();
      let subgraphlist = summary[i].split(':')[1].replace(/ /g, '').replace('{', '').replace('}', '').replace(/,/g, ' ');
      let tmp;

      if(backend.indexOf('webnn') >-1) {
        tmp = `<li><div class="timeline-badge tb-webnn"><i class="glyphicon">WebNN</i></div><div class="timeline-panel tp-webnn"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      } else if (backend.indexOf('wasm') >-1) {
        tmp = `<li class="timeline-inverted"><div class="timeline-badge tb-wasm"><i class="glyphicon">WASM</i></div><div class="timeline-panel tp-wasm"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      } else if (backend.indexOf('webgl') >-1) {
        tmp = `<li class="timeline-inverted"><div class="timeline-badge tb-webgl"><i class="glyphicon">WebGL</i></div><div class="timeline-panel tp-webgl"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      }

      listhtml += tmp;
    }
    $('#subgraph').html(listhtml);
  }
}

const setPreferenceCodeToolTip = () => {
  if($('#backendpolyfilltitle')) {
    $('#backendpolyfilltitle').attr('data-html', 'true')
    .attr('data-placement', 'bottom')
    .attr('title',
      `<div class="backendtooltip">WASM: Compiled Tensorflow Lite C++ kernels to WebAssembly format.<br>
      WebGL: Tensorflow.js WebGL kernel.</div>`
    );
    $('#backendpolyfilltitle').tooltip();
  }
  if($('#backendwebnntitle')) {
    $('#backendwebnntitle').attr('data-html', 'true')
    .attr('data-placement', 'bottom')
    .attr('title',
      `<div class="backendtooltip">FAST_SINGLE_ANSWER: Prefer returning a single answer as fast as possible, even if this causes more power consumption.<br>
      SUSTAINED_SPEED: Prefer maximizing the throughput of successive frames, for example when processing successive frames coming from the camera.<br>
      LOW_POWER: Prefer executing in a way that minimizes battery drain. This is desirable for compilations that will be executed often.</div>`
    );
    $('#backendwebnntitle').tooltip();
  }
}

const updateTitle = (name, backend, prefer, model, modeltype) => {
  model = model.replace(/_/g, ' ');
  let currentprefertext = {
    // fast: 'FAST_SINGLE_ANSWER',
    // sustained: 'SUSTAINED_SPEED',
    // low: 'LOW_POWER',
    fast: 'FAST',
    sustained: 'SUSTAINED',
    low: 'LOW',   
    none: 'None',
  }[prefer];

  let backendtext = backend;
  if (backendtext == 'WebML') {
    backendtext = 'WebNN';
  }
  if (backend !== 'WebML' && prefer !== 'none') {
    backendtext = backend + ' + WebNN';
  }

  if(currentprefertext === 'None') {
    $('#ictitle').html(`${name} / ${backendtext} / ${model} (${modeltype})`);
  } else {
    $('#ictitle').html(`${name} / ${backendtext} (${currentprefertext}) / ${model} (${modeltype})`);
  }
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
  if(navigator.userAgent.toLowerCase().indexOf("edge") > -1) {
    if(location.pathname.toLocaleLowerCase() === '/examples/' || location.pathname.toLocaleLowerCase().indexOf('/examples/index') >-1 || location.pathname.toLocaleLowerCase().indexOf('/examples/model') >-1) {
      $('#logo').html('<img src="static/img/edge_logo.png">')
    } else {
      $('#logo').html('<img src="../static/img/edge_logo.png">')
    }
  }

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

  setPreferenceCodeToolTip();

});

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

const updateBackendRadioUI = (backend, prefer) => {
  let polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
  let webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');
  if (backend !== 'none' && backend.toLocaleLowerCase() !== 'webml' && prefer !== 'none') {
    $('.backend label').removeClass('x');
    $('#l-' + polyfillId).addClass('x');
    $('#l-' + webnnId).addClass('x');
    $('.backendtitle').html('Backends');
    $('#backendswitch').prop('checked', true);
  } else {
    $('.backend label').removeClass('x');
    $('.backendtitle').html('Backend');
    $('#backendswitch').prop('checked', false);
  }
}

let isBackendSwitch = () => {
  return $('#backendswitch').is(':checked')
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

  updateBackendRadioUI(ub, up);
});


if (skeletonDetectionPath <= -1) {
  $(document).ready(() => {

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
        updateTitle('Sole Backend', currentBackend, currentPrefer, `${um}`, `${ut}`);
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
        updateTitle('Dual Backends', currentBackend, currentPrefer, `${um}`, `${ut}`);
      }

      updateBackendRadioUI(currentBackend, currentPrefer);
      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);
      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }
      updateBackend(us === 'camera', true);
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

      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);
      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }
      updateBackend(us === 'camera', true);
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
      strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
      window.history.pushState(null, null, strsearch);
      if (um === 'none') {
        showError('No model selected', 'Please select a model to start prediction.');
        return;
      }
      updateBackend(us === 'camera', true);
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
  $('#progressstep').html(text);
  $('.shoulddisplay').hide();
  $('.icdisplay').hide();
  $('#resulterror').hide();
  await new Promise(res => setTimeout(res, 100));
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