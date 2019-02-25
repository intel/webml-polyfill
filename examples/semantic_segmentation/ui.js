let up = getUrlParam('prefer');
let ub = getUrlParam('b');
let um = getUrlParam('m');
let ut = getUrlParam('t');
let us = getUrlParam('s');
let ud = getUrlParam('d');
let strsearch;

if (!location.search) {
  strsearch = `?prefer=none&b=WASM&m=none&t=none&s=image&d=0`;
  let path = location.href;
  location.href = path + strsearch;
}

const componentToggle = () => {
  // $('#header-sticky-wrapper').attr('style', 'display:block');
  $('#header-sticky-wrapper').slideToggle();
  $('#query').slideToggle();
  $('.nav-pills').slideToggle();
  $('.github-corner').slideToggle();
  // $('#mobile-nav-toggle').slideToggle(100);
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

const buttonUI = (camera = false) => {
  if(camera) {
    $('#pickimage').hide();
    $('#fps').show();
  } else {
    $('#pickimage').show();
    $('#fps').hide();
  }
}

const setFullScreenIconPosition = (modelname) => {
  let svgstyle = 'p' + modelname.replace('deeplab_mobilenet_v2_', '').replace('_tflite', '').replace(/_/g, '').replace('atrous','');
  $('#semanticsegmentation #fullscreen i svg').removeClass('p224').removeClass('p257').removeClass('p321').removeClass('p513').addClass(svgstyle);
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
    setFullScreenIconPosition(um);
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

  const updateTitle = (backend, prefer, model, modeltype) => {
    model = model.replace('mobilenet', '').replace('v2', '').replace(/_/g, ' ');
    let currentprefertext;
    if (backend == 'WASM' || backend == 'WebGL') {
      $('#ictitle').html(`Semantic Segmentation / ${backend} / ${model} (${modeltype})`);
    } else if (backend == 'WebML') {
      if (getUrlParam('p') == 'fast') {
        prefer = 'FAST_SINGLE_ANSWER';
      } else if (getUrlParam('p') == 'sustained') {
        prefer = 'SUSTAINED_SPEED';
      } else if (getUrlParam('p') == 'low') {
        prefer = 'LOW_POWER';
      }
      $('#ictitle').html(`Semantic Segmentation / WebNN / ${prefer} / ${model} (${modeltype})`);
    }
  }
  updateTitle(ub, up, um, ut);

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

    updateTitle(currentBackend, currentPrefer, `${um}`, `${ut}`);
    strsearch = `?prefer=${currentPrefer}&b=${currentBackend}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if (um === 'none') {
      showError('No model selected', 'Please select a model to start prediction.');
      return;
    }
    updateScenario(us == 'camera');
  });

  $('input:radio[name=m]').click(() => {
    $('.alert').hide();
    let rid = $('input:radio[name="m"]:checked').attr('id');
    if (rid.indexOf('_onnx') > -1) {
      um = rid.replace('_onnx', '');
      ut = 'onnx';
    }
    if (rid.indexOf('_tflite') > -1) {
      um = rid.replace('_tflite', '');
      ut = 'tflite';
    }

    if (rid.indexOf('_tflite') > -1) {
      um = rid.replace('_tflite', '');
      ut = 'tflite';
    }

    setFullScreenIconPosition(rid);

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
    updateTitle(currentBackend, currentPrefer, `${um}`, `${ut}`);
    main(us == 'camera');
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
    strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if (um === 'none') {
      showError('No model selected', 'Please select a model to start prediction.');
      return;
    }

    updateScenario(false);
    buttonUI(us === 'camera');
  });

  $('#cam').click(() => {
    $('.alert').hide();
    $('ul.nav-pills li').removeClass('active');
    $('ul.nav-pills #cam').addClass('active');
    $('#cameratab').addClass('active');
    $('#imagetab').removeClass('active');
    us = 'camera';
    strsearch = `?prefer=${up}&b=${ub}&m=${um}&t=${ut}&s=${us}&d=${ud}`;
    window.history.pushState(null, null, strsearch);

    if (um === 'none') {
      showError('No model selected', 'Please select a model to start prediction.');
      return;
    }

    updateScenario(true);
    buttonUI(us === 'camera');
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
    $('.zoom-wrapper').toggle();
    $('#labelitem').toggle();
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
  $('.icdisplay').show();
  $('.shoulddisplay').show();
  $('#resulterror').hide();
  buttonUI(us === 'camera');
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

const zoomSlider = document.getElementById('zoomSlider');
const blurSlider = document.getElementById('blurSlider');
const refineEdgeSlider = document.getElementById('refineEdgeSlider');
const colorMapAlphaSlider = document.getElementById('colorMapAlphaSlider');
const selectBackgroundButton = document.getElementById('chooseBackground');
const clearBackgroundButton = document.getElementById('clearBackground');

$(window).load(() => {

  let colorPicker = new iro.ColorPicker('#color-picker-container', {
    width: 200,
    height: 200,
    color: {
      r: renderer.bgColor[0],
      g: renderer.bgColor[1],
      b: renderer.bgColor[2]
    },
    markerRadius: 5,
    sliderMargin: 12,
    sliderHeight: 20,
  });

  $('.bg-value').html(colorPicker.color.hexString);

  colorPicker.on('color:change', function (color) {
    $('.bg-value').html(color.hexString);
    renderer.bgColor = [color.rgb.r, color.rgb.g, color.rgb.b];
  });

  zoomSlider.value = renderer.zoom * 100;
  $('.zoom-value').html(renderer.zoom + 'x');
  zoomSlider.oninput = () => {
    let zoom = zoomSlider.value / 100;
    $('.zoom-value').html(zoom + 'x');
    renderer.zoom = zoom;
  };

  colorMapAlphaSlider.value = renderer.colorMapAlpha * 100;
  $('.color-map-alpha-value').html(renderer.colorMapAlpha);
  colorMapAlphaSlider.oninput = () => {
    let alpha = colorMapAlphaSlider.value / 100;
    $('.color-map-alpha-value').html(alpha);
    renderer.colorMapAlpha = alpha;
  };

  blurSlider.value = renderer.blurRadius;
  $('.blur-radius-value').html(renderer.blurRadius + 'px');
  blurSlider.oninput = () => {
    let blurRadius = parseInt(blurSlider.value);
    $('.blur-radius-value').html(blurRadius + 'px');
    renderer.blurRadius = blurRadius;
  };

  refineEdgeSlider.value = renderer.refineEdgeRadius;
  if (refineEdgeSlider.value === '0') {
    $('.refine-edge-value').html('DISABLED');
  } else {
    $('.refine-edge-value').html(refineEdgeSlider.value + 'px');
  }
  refineEdgeSlider.oninput = () => {
    let refineEdgeRadius = parseInt(refineEdgeSlider.value);
    if (refineEdgeRadius === 0) {
      $('.refine-edge-value').html('DISABLED');
    } else {
      $('.refine-edge-value').html(refineEdgeRadius + 'px');
    }
    renderer.refineEdgeRadius = refineEdgeRadius;
  };

  $('.effects-select .btn input').filter(function () {
    return this.value === renderer.effect;
  }).parent().toggleClass('active');

  $('.controls').attr('data-select', renderer.effect);

  $('.effects-select .btn').click((e) => {
    e.preventDefault();
    let effect = e.target.children[0].value;
    $('.controls').attr('data-select', effect);
    renderer.effect = effect;
  });

  selectBackgroundButton.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      let img = new Image();
      img.onload = function () {
        renderer.backgroundImageSource = img;
      };
      img.src = URL.createObjectURL(files[0]);
    }
  }, false);

  clearBackgroundButton.addEventListener('click', (e) => {
    renderer.backgroundImageSource = null;
  }, false);

  function getMousePos(canvas, evt) {
    let rect = canvas.getBoundingClientRect();
    return {
      x: Math.ceil(evt.clientX - rect.left),
      y: Math.ceil(evt.clientY - rect.top)
    };
  }

  outputCanvas.addEventListener('mousemove', (e) => {
    hoverPos = getMousePos(outputCanvas, e);
    renderer.highlightHoverLabel(hoverPos);
  });
  outputCanvas.addEventListener('mouseleave', (e) => {
    hoverPos = null;
    renderer.highlightHoverLabel(hoverPos);
  });

});

$(window).load(() => {
  if (ud != '0') {
    componentToggle();
  }
  disableModel();
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})