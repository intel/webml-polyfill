let up = getUrlParam('prefer');
let ub = getUrlParam('b');
let um = getUrlParam('m');
let ut = getUrlParam('t');
let us = getUrlParam('s');
let ud = getUrlParam('d');
let currenttab = getUrlParam('s');
let strsearch;

if (!location.search) {
  strsearch = `?prefer=none&b=WASM&m=mobilenet_v1&t=tflite&s=image&d=0`;
  let path = location.href;
  location.href = path + strsearch;
}

function componentToggle() {
  // $('#header-sticky-wrapper').attr('style', 'display:block');
  $('#header-sticky-wrapper').slideToggle();
  $('#query').slideToggle();
  $('.nav-pills').slideToggle();
  $('.github-corner').slideToggle();
  // $('#mobile-nav-toggle').slideToggle(100);
  $('footer').slideToggle();
  $('#extra span').toggle();
}

$(document).ready(function () {

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
  }

  if (hasUrlParam('b')) {
    $('.backend input').removeAttr('checked');
    $('.backend label').removeClass('checked');
    $('#' + getUrlParam('b')).attr('checked', 'checked');
    $('#l-' + getUrlParam('b')).addClass('checked');
  }

  if (hasUrlParam('m') && hasUrlParam('t')) {
    $('.model input').removeAttr('checked');
    $('.model label').removeClass('checked');
    let m_t = getUrlParam('m') + '_' + getUrlParam('t');
    $('#' + m_t).attr('checked', 'checked');
    $('#l-' + m_t).addClass('checked');
  }

  if (hasUrlParam('prefer')) {
    $('.prefer input').removeAttr('checked');
    $('.prefer label').removeClass('checked');
    $('#' + getUrlParam('prefer')).attr('checked', 'checked');
    $('#l-' + getUrlParam('prefer')).addClass('checked');

    if(ub == 'WASM' || ub == 'WebGL') {
      $('.ml').removeAttr('checked');
      $('.lml').removeClass('checked');
    }
  }

  function updateTitle(backend, prefer, model, modeltype) {
    let currentprefertext;
    if (backend == 'WASM' || backend == 'WebGL') {
      $('#ictitle').html(`Image Classfication / ${backend} / ${model} (${modeltype})`);
    } else if (backend == 'WebML') {
      if (getUrlParam('p') == 'fast') {
        prefer = 'FAST_SINGLE_ANSWER';
      } else if (getUrlParam('p') == 'sustained') {
        prefer = 'SUSTAINED_SPEED';
      } else if (getUrlParam('p') == 'low') {
        prefer = 'LOW_POWER';
      }
      $('#ictitle').html(`Image Classfication / ${backend} / ${prefer} / ${model} (${modeltype})`);
    }
  }
  updateTitle(ub, up, um, ut);

  $('input:radio[name=b]').click(function () {
    $('.alert').hide();
    let rid = $("input:radio[name='b']:checked").attr('id');
    $('.backend input').removeAttr('checked');
    $('.backend label').removeClass('checked');
    $('#' + rid).attr('checked', 'checked');
    $('#l-' + rid).addClass('checked');

    if(rid == 'WASM' || rid == 'WebGL') {
      $('.ml').removeAttr('checked');
      $('.lml').removeClass('checked');
    }

    if(rid == 'WASM' || rid == 'WebGL') {
      currentBackend = rid;
      currentPrefer = 'none';
    } else if (rid == 'fast' || rid == 'sustained' || rid == 'low') {
      currentBackend = 'WebML';
      currentPrefer = rid;
    }

    updateTitle(currentBackend, currentPrefer, `${um}`, `${ut}`);

    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=image&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if (currenttab == 'camera') {
      updateScenario(true, currentBackend, currentPrefer);
    } else {
      updateScenario(false, currentBackend, currentPrefer);
    }
  });

  $('input:radio[name=m]').click(function () {
    $('.alert').hide();
    let rid = $("input:radio[name='m']:checked").attr('id');
    if (rid.indexOf('_onnx') > -1) {
      um = rid.replace('_onnx', '');
      ut = 'onnx';
    }
    if (rid.indexOf('_tflite') > -1) {
      um = rid.replace('_tflite', '');
      ut = 'tflite';
    }
    if(currentBackend && currentPrefer) {
      strsearch= `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
    } else {
      strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
    }
    // location.href = strsearch;
    window.history.pushState(null, null, strsearch);
    

    if (`${um}` && `${ut}`) {
      $('.model input').removeAttr('checked');
      $('.model label').removeClass('checked');
      let m_t = `${um}` + '_' + `${ut}`;
      $('#' + m_t).attr('checked', 'checked');
      $('#l-' + m_t).addClass('checked');
    }

    currentModel = `${um}_${ut}`;
    updateTitle(currentBackend, currentPrefer, `${um}`, `${ut}`);
    (us == 'camera') ? main(true) : main();
  });

  $('#extra').click(function () {
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
    if(currentBackend && currentPrefer) {
      strsearch= `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${display}`;
    } else {
      strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${display}`;
    }
    window.history.pushState(null, null, strsearch);
  });
});

$(document).ready(function () {
  $('#img').click(function () {
    $('.alert').hide();
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #img').addClass('active');
    $('#imagetab').addClass('active');
    $('#cameratab').removeClass('active');
    strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=image&d=${ud}`;
    window.history.pushState(null, null, strsearch)
    currenttab = 'image';
    updateScenario(false, currentBackend, currentPrefer);
  });

  $('#cam').click(function () {
    $('.alert').hide();
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #cam').addClass('active');
    $('#cameratab').addClass('active');
    $('#imagetab').removeClass('active');
    strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=camera&d=${ud}`;
    window.history.pushState(null, null, strsearch)
    currenttab = 'camera';
    updateScenario(true, currentBackend, currentPrefer);
  });

  $('#fullscreen i svg').click(function () {
    $('#fullscreen i').toggle();
    toggleFullScreen();
    $('video').toggleClass('fullscreen');
    $('#overlay').toggleClass('video-overlay');
    $('#fps').toggleClass('fullscreen');
    $('#fullscreen i').toggleClass('fullscreen');
    $('#ictitle').toggleClass('fullscreen');
    $('#inference').toggleClass('fullscreen');
  });

});

$(document).ready(function () {
  $('.model label').each(function () {
    let timeoutObj = null;
    $(this).on("mouseenter mouseleave touchstart touchcancel touchend", function (e) {
      let _this = this;
      if (e.type == "mouseenter" || e.type == "touchstart") {
        (function () {
          timeoutObj = setTimeout(function () {
            let modelid = _this.id.replace('l-', '');
            for (model of imageClassificationModels) {
              if (modelid == model.modelName) {
                $('#intro').slideDown();
                if (model.intro) {
                  $('#introdescription').html(model.intro);
                  $('#introdescription').removeClass('dnone');
                } else {
                  $('#introdescription').addClass('dnone');
                }

                if (model.paperUrl) {
                  $('#paperurl').html('Paper');
                  $('#paperurl').attr('href', model.paperUrl);
                  $('#paperurl').removeClass('dnone');
                } else {
                  $('#paperurl').addClass('dnone');
                }

                if (model.modelCDNUrl) {
                  $('#netronurl').html(model.modelName + ' Model Viewer');
                  $('#netronurl').attr('href', `https://lutzroeder.github.io/netron/?url=${model.modelCDNUrl}`);
                  $('#netronurl').removeClass('dnone');
                } else {
                  $('#netronurl').addClass('dnone');
                }
              }
            }
          }, 3000);
        }(this));
      } else if (e.type == "mouseleave" || e.type == "touchcancel" || e.type == "touchend") {
        if (timeoutObj != null) {
          clearTimeout(timeoutObj);
          $('#intro').delay(1000).slideUp();
        }
      }
    });
  })
});

$(window).load(function () {
  if (ud != '0') {
    componentToggle();
  }
});

async function showProgress(text) {
  $('#progressmodel').show();
  await $('#progressstep').html(text);
  $('.shoulddisplay').hide();
  $('.icdisplay').hide();
  $('#resulterror').hide();
}

function showResults() {
  $('#progressmodel').hide();
  $('.icdisplay').fadeIn();
  $('.shoulddisplay').fadeIn();
  $('#resulterror').hide();
}

function showError() {
  $('#progressmodel').hide();
  $('.icdisplay').hide();
  $('.shoulddisplay').hide();
  $('#resulterror').fadeIn();
}

function updateLoading(c) {
  $(".loading-page .counter h1").html(c + "%");
}

$(window).load(function () {
  (us == 'camera') ? main(true) : main();
})