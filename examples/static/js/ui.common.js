// Common code for examples part including index.html, model.html and concret example's xxx/index.hml, likes image_classification/index.html etc.
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
};

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

const updateSIMDNotes = () => {
  const searchParams = new URLSearchParams(location.search);
  const f = searchParams.get('f');
  if (f && f.toLowerCase() === 'opencv.js') {
    $('#simdnotes').html(`Please enable following flags to experience the experimental features Threads and SIMD with Google Chrome browser.
      <ol>
        <li>Type <a href="chrome://flags">chrome://flags</a> in URL address bar and press "Enter" key</li>
        <li>Search "WebAssembly threads support" and "WebAssembly SIMD support"</li>
        <li>Select "Enabled", relaunch browser</li>
      </ol>`).show();
  } else {
    $('#simdnotes').hide();
  }
}

$(document).ready(() => {
  $('#header').sticky({ topSpacing: 0, zIndex: '50' });

  $('.nav-menu').superfish({
    animation: { opacity: 'show' },
    speed: 400
  });

  if ($('#nav-container').length > 0) {
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

// Below Common code for concret example's xxx/index.hml, likes image_classification/index.html etc.
const formatToLogo = {
  'tensorflow': '../static/img/l-tflite.png',
  'tflite': '../static/img/l-tflite.png',
  'onnx': '../static/img/l-onnx.png',
  'openvino': '../static/img/l-openvino.png',
  'caffe2': '../static/img/l-caffe2.png',
};

const trademarks = (allFormats) => {
  let trademarknote = '';

  for (let format of allFormats) {
    let trademark = '';
    switch (format.toLowerCase()) {
      case 'tensorflow':
      case 'tflite':
        trademark = 'TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc. ';
        break;
      case 'onnx':
        trademark += 'ONNX is a community project created by Facebook and Microsoft. ONNX is a trademark of Facebook, Inc. ';
        break;
      case 'openvino':
        trademark += 'OpenVINO and the OpenVINO logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries. ';
        break;
      case 'caffe2':
        trademark += 'Caffe2 is now a part of PyTorch. ';
        break;
      default:
        break;
    }
    trademarknote += trademark;
  }

  if (trademarknote) {
    $('#trademark').html(trademarknote);
  }
};

updateSIMDNotes();

const singleModelTable = (modelList, category) => {
  const allFormats = new Set(modelList.map((m) => m.format));
  const backendTr = $('.backend');
  const trows = [];
  for (const format of allFormats) {
    let trow = $(`<tr class='model' id='${category}'>`);
    let tdata = $('<td>');
    trow.append(tdata);
    const logo = formatToLogo[format.toLowerCase()];
    tdata.append($(`<img src='${logo}' alt='${format} Format' title='${format} Format'>`));
    const models = modelList.filter((m) => format === m.format && !m.disabled);
    for (const model of models) {
      const modelName = model.modelName.replace(/ \(.*\)$/, '');
      const modelId = model.modelId;
      tdata.append($(`<input type='radio' class='d-none' name='m' id='${modelId}' value='${modelId}'>`));
      tdata.append($(`<label id='l-${modelId}' class='themodel' for='${modelId}' title='${modelName}'>${modelName}</label>`));
    }
    trows.push(trow);
  }
  trows[0].prepend($(`<th class='text-center' rowspan='${allFormats.size}'>
                        <a href='../model.html' title='View model details'>${category}</a>
                      </th>`));
  backendTr.before(trows);
  return allFormats;
};

const setModelComponents = (models, selectedModelIdStr) => {
  $('.model').remove();
  let formatTypes = [];

  for (let [category, modelList] of Object.entries(models)) {
    let formats = singleModelTable(modelList, category);
    formatTypes.push(...formats);
    formatTypes = [...new Set(formatTypes)];
  }

  trademarks(formatTypes);
  updateModelComponentsStyle(selectedModelIdStr);
};

const updateFrameworkComponentsStyle = (framework) => {
  const _framework = framework.replace('.', '');
  $('.framework input').attr('disabled', false);
  $('.framework label').removeClass('cursordefault');
  $('#framework' + _framework).attr('disabled', true);
  $('#l-framework' + _framework).addClass('cursordefault');
  $('.framework input').removeAttr('checked');
  $('.framework label').removeClass('checked');
  $('#framework' + _framework).attr('checked', 'checked');
  $('#l-framework' + _framework).addClass('checked');
};

const setFrameworkComponents = (frameworkList, selectedFramework) => {
  const tbody = $('#query tbody');
  let trow = $(`<tr class='framework'>`);
  trow.append(`<th class='text-center'>Framework</th>`);
  let tdata = $('<td>');
  trow.append(tdata);
  for (const framework of frameworkList) {
    const _framework = framework.replace('.', '');
    tdata.append($(`<input type='radio' class='d-none' name='framework' id='framework${_framework}' value='${framework}'>`));
    tdata.append($(`<label id='l-framework${_framework}' for='framework${_framework}'>${framework}</label>`));
  }
  tbody.prepend(trow);
  updateFrameworkComponentsStyle(selectedFramework);
};

const updateOpenCVJSBackendComponentsStyle = (selectedBackend) => {
  const _selectedBackend = selectedBackend.toLocaleLowerCase().replace(' ', '');
  $('.opencvjsbackend input').attr('disabled', false);
  $('.opencvjsbackend label').removeClass('cursordefault');
  $('#opencvjs' + _selectedBackend).attr('disabled', true);
  $('#l-opencvjs' + _selectedBackend).addClass('cursordefault');
  $('.opencvjsbackend input').removeAttr('checked');
  $('.opencvjsbackend label').removeClass('checked');
  $('#opencvjs' + _selectedBackend).attr('checked', 'checked');
  $('#l-opencvjs' + _selectedBackend).addClass('checked');
};

const updateOpenVINOJSBackendComponentsStyle = (selectedBackend) => {
  const _selectedBackend = selectedBackend.toLocaleLowerCase().replace(' ', '');
  $('.openvinojsbackend input').attr('disabled', false);
  $('.openvinojsbackend label').removeClass('cursordefault');
  $('#openvinojsbackend' + _selectedBackend).attr('disabled', true);
  $('#l-openvinojsbackend' + _selectedBackend).addClass('cursordefault');
  $('.openvinojsbackend input').removeAttr('checked');
  $('.openvinojsbackend label').removeClass('checked');
  $('#openvinojs' + _selectedBackend).attr('checked', 'checked');
  $('#l-openvinojs' + _selectedBackend).addClass('checked');
}

const setPreferenceTipComponents = () => {
  if ($('#backendpolyfilltitle')) {
    $('#backendpolyfilltitle').attr('data-html', 'true')
      .attr('data-placement', 'bottom')
      .attr('title',
        `<div class="backendtooltip">WASM: TensorFlow.js WebAssembly backend builds on top of the XNNPACK library.<br>
      WebGL: TensorFlow.js GPU accelerated WebGL backend.</div>`
      );
    $('#backendpolyfilltitle').tooltip();
  }

  if ($('#backendwebnntitle')) {
    $('#backendwebnntitle').attr('data-html', 'true')
      .attr('data-placement', 'bottom')
      .attr('title',
        `<div class="backendtooltip">FAST_SINGLE_ANSWER: Prefer returning a single answer as fast as possible, even if this causes more power consumption.<br>
      SUSTAINED_SPEED: Prefer maximizing the throughput of successive frames, for example when processing successive frames coming from the camera.<br>
      LOW_POWER: Prefer executing in a way that minimizes battery drain. This is desirable for compilations that will be executed often.</div>`
      );
    $('#backendwebnntitle').tooltip();
  }
};

const isBackendSwitch = () => {
  return $('#backendswitch').is(':checked')
};

const isFrontFacingSwitch = () => {
  return $('#cameraswitch').is(':checked')
};

const getModelListByClass = () => {
  let ids = [];
  for (let model of $('#query tbody .model')) {
    ids.push(model.id);
  }
  return [...new Set(ids)];
};

const showAlertComponent = (error) => {
  console.error(error);
  $('#progressmodel').hide();
  $('.icdisplay').hide();
  $('.shoulddisplay').hide();
  $('#resulterror').fadeIn();
  let div = document.createElement('div');
  div.setAttribute('class', 'backendAlert alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = `<strong>${error.message}</strong>`;
  div.innerHTML += `<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button>`;
  let container = document.getElementById('container');
  container.insertBefore(div, container.firstElementChild);
};

const showErrorComponent = (title = null, description = null) => {
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
};

const updateModelComponentsStyle = (modelIdStr) => {
  if (modelIdStr !== null) {
    let modelId = null;
    if (modelIdStr.includes('+')) {
      let umArray = modelIdStr.split('+');
      for (modelId of umArray) {
        // reset style
        let modelClass = $('#' + modelId).parent().parent().attr('id');
        $('.model[id=' + modelClass + '] input').attr('disabled', false);
        $('.model[id=' + modelClass + '] label').removeClass('cursordefault');
        $('.model[id=' + modelClass + '] input').removeAttr('checked');
        $('.model[id=' + modelClass + '] label').removeClass('checked');
        $('#' + modelId).attr('disabled', true);
        $('#l-' + modelId).addClass('cursordefault');
        $('#' + modelId).attr('checked', 'checked');
        $('#l-' + modelId).addClass('checked');
      }
    } else if (modelIdStr.includes(' ')) {
      let umArray = modelIdStr.split(' ');
      for (modelId of umArray) {
        // reset style
        let modelClass = $('#' + modelId).parent().parent().attr('id');
        $('.model[id=' + modelClass + '] input').attr('disabled', false);
        $('.model[id=' + modelClass + '] label').removeClass('cursordefault');
        $('.model[id=' + modelClass + '] input').removeAttr('checked');
        $('.model[id=' + modelClass + '] label').removeClass('checked');
        $('#' + modelId).attr('disabled', true);
        $('#l-' + modelId).addClass('cursordefault');
        $('#' + modelId).attr('checked', 'checked');
        $('#l-' + modelId).addClass('checked');
      }
    } else {
      $('.model input').attr('disabled', false);
      $('.model label').removeClass('cursordefault');
      $('#' + modelIdStr).attr('disabled', true);
      $('#l-' + modelIdStr).addClass('cursordefault');
      $('.model input').removeAttr('checked');
      $('.model label').removeClass('checked');
      $('#' + modelIdStr).attr('checked', 'checked');
      $('#l-' + modelIdStr).addClass('checked');
    }
  }
};

const updateBackendComponents = (backend, prefer) => {
  const polyfillId = $('input:radio[name="bp"]:checked').attr('id') || $('input:radio[name="bp"][checked="checked"]').attr('id');
  const webnnId = $('input:radio[name="bw"]:checked').attr('id') || $('input:radio[name="bw"][checked="checked"]').attr('id');

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

  // Required from feedback
  if (prefer === 'sustained' & currentOS === 'Mac OS') {
    $('#l-sustained').html('SUSTAINED_SPEED');
    $('#l-sustained').append(`<span class='nbackend'>MPS</span>`);
  } else {
    $('#l-sustained').html('SUSTAINED_SPEED');
  }
};

const updateTitleComponent = (backend, prefer, modelIdStr = null, modelInfoDic = null) => {
  
  const getExampleName = () => {
    const pathArray = location.pathname.split('/');
    const dirName = pathArray[pathArray.length - 2];
    return dirName.replace(/_/g, ' ');
  };

  let sampleName = getExampleName();
  if(sampleName.indexOf('opencv') > -1) {
    sampleName = sampleName.replace('opencv', '') + 'OpenCV'
  }

  let modelShow = null;
  if (modelIdStr != null) {
    if (modelIdStr != 'none') {
      let modelIdArray;
      if (modelIdStr.includes('+') || modelIdStr.includes(' ')) {
        if (modelIdStr.includes('+')) {
          modelIdArray = modelIdStr.split('+');
        } else if (modelIdStr.includes(' ')) {
          modelIdArray = modelIdStr.split(' ');
        }
        for (let modelId of modelIdArray) {
          let targetModelInfo = getModelFromGivenModels(modelInfoDic, modelId);
          if (modelShow === null) {
            modelShow = getModelFromGivenModels(modelInfoDic, modelId).modelName;
          } else {
            modelShow = modelShow + ' + ' + getModelFromGivenModels(modelInfoDic, modelId).modelName;
          }
        }
      } else {
        let targetModelInfo = getModelFromGivenModels(modelInfoDic, modelIdStr);
        modelShow = targetModelInfo.modelName;
      }
    } else {
      modelShow = 'None';
    }
  }

  if (prefer != null) {
    let currentPreferText = {fast: 'FAST',
    sustained: 'SUSTAINED',
    low: 'LOW',
    ultra_low: 'ULTRA_LOW',
    none: 'None',}[prefer];
    let backendText = backend;
    if (backendText == 'WebML') {
      backendText = 'WebNN';
    }
    if (backend !== 'WebML' && prefer !== 'none') {
      backendText = backend + ' + WebNN';
    }

    if (modelIdStr != null) {
      if (currentPreferText === 'None') {
        $('#ictitle').html(`${sampleName} / ${backendText} / ${modelShow}`);
      } else if(prefer === 'sustained' & currentOS === 'Mac OS') {
        $('#ictitle').html(`${sampleName} / ${backendText} (${currentPreferText}/MPS) / ${modelShow}`);
      } else {
        $('#ictitle').html(`${sampleName} / ${backendText} (${currentPreferText}) / ${modelShow}`);
      }
    } else {
      if (currentPreferText === 'None') {
        $('#ictitle').html(`${sampleName} / ${backendText}`);
      } else if(prefer === 'sustained' & currentOS === 'Mac OS') {
        $('#ictitle').html(`${sampleName} / ${backendText} (${currentPreferText}/MPS)`);
      } else {
        $('#ictitle').html(`${sampleName} / ${backendText} (${currentPreferText})`);
      }
    }
  } else {
    let _backend = backend.replace(' ', '+');
    $('#ictitle').html(`${sampleName} / ${_backend} / ${modelShow}`);
  }
};

const showProgressComponent = async (pm, pb, pi) => {
  let p = '';
  let modelicon = ``;
  if (pm === 'done') {
    modelicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pm === 'current') {
    modelicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    modelicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  let updateicon = ``;
  if (pb === 'done') {
    updateicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pb === 'current') {
    updateicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    updateicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  let inferenceicon = ``;
  if (pi === 'done') {
    inferenceicon = `<svg class='prog_list_icon' viewbox='0 0 24 24'>
                    <path class='st0' d='M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8zm0 1.5c-5.2 0-9.5-4.3-9.5-9.5S6.8 2.5 12 2.5s9.5 4.3 9.5 9.5-4.3 9.5-9.5 9.5z'></path>
                    <path class='st0' d='M11.1 12.9l-1.2-1.1c-.4-.3-.9-.3-1.3 0-.3.3-.4.8-.1 1.1l.1.1 1.8 1.6c.1.1.4.3.7.3.2 0 .5-.1.7-.3l3.6-4.1c.3-.3.4-.8.1-1.1l-.1-.1c-.4-.3-1-.3-1.3 0l-3 3.6z'></path>
                  </svg>`;
  } else if (pi === 'current') {
    inferenceicon = `<svg class='prog_list_icon prog_list_icon-${pb}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  } else {
    inferenceicon = `<svg class='prog_list_icon prog_list_icon-${pi}' width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12 16.1c1.8 0 3.3-1.4 3.3-3.2 0-1.8-1.5-3.2-3.3-3.2s-3.3 1.4-3.3 3.2c0 1.7 1.5 3.2 3.3 3.2zm0 1.7c-2.8 0-5-2.2-5-4.9S9.2 8 12 8s5 2.2 5 4.9-2.2 4.9-5 4.9z'></path>
                </svg>`;
  }

  p = `
      <nav class='prog'>
        <ul class='prog_list'>
          <li class='prog prog-${pm}'>
            ${modelicon}<span class='prog_list_title'>Model loading</span>
          </li>
          <li class='prog prog-${pb}'>
            ${updateicon}<span class='prog_list_title'>Model compilation</span>
          </li>
          <li class='prog prog-${pi}'>
            ${inferenceicon}<span class='prog_list_title'>Model inferencing</span>
          </li>
        </ul>
      </nav>
    `;

  $('#progressmodel').show();
  $('#progressstep').html(p);
  $('.shoulddisplay').hide();
  $('.icdisplay').hide();
  $('#resulterror').hide();
  await new Promise(res => setTimeout(res, 100));
};

const showOpenCVRuntimeProgressComponent = async () => {
  let modelicon = `<svg width='24' height='24' viewbox='0 0 24 24'>
                  <path d='M12.2 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16zm0 1.377a9.377 9.377 0 1 1 0-18.754 9.377 9.377 0 0 1 0 18.754zm-4-8a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754zm4 0a1.377 1.377 0 1 1 0-2.754 1.377 1.377 0 0 1 0 2.754z' fill='#006DFF' fill-rule='evenodd'></path>
                </svg>`;
  let p = `
      <nav class='prog'>
        <ul class='prog_list'>
          <li>
            ${modelicon}<span class='prog_list_title'>Loading OpenCV Runtime</span>
          </li>
        </ul>
      </nav>
    `;
  $('#progressruntime').show();
  await new Promise(res => setTimeout(res, 100));
};

const updateLoadingProgressComponent = (ev) => {
  if (ev.lengthComputable) {
    let totalSize = ev.total / (1000 * 1000);
    let loadedSize = ev.loaded / (1000 * 1000);
    let percentComplete = ev.loaded / ev.total * 100;
    percentComplete = percentComplete.toFixed(0);
    const progressBar = document.getElementById('progressBar');
    progressBar.style = `width: ${percentComplete}%`;
    $('.loading-page .counter h1').html(`${loadedSize.toFixed(1)}/${totalSize.toFixed(1)}MB ${percentComplete}%`);
  }
};

const readyShowResultComponents = () => {
  $('#progressmodel').hide();
  $('.icdisplay').fadeIn();
  $('.shoulddisplay').fadeIn();
  $('#resulterror').hide();
};

const showHybridComponent = (supportedOps, requiredOps, backend, prefer, multi = false) => {
  const hybridRow = (offloadops, backend, prefer) => {
    if (offloadops && offloadops.size > 0 && backend != 'WebML' && prefer != 'none') {
      $('.offload').fadeIn();
      let offloadopsvalue = '';
      offloadops.forEach((value) => {
        let t = '<span class="ol">' + operationTypes[value] + '</span>';
        offloadopsvalue += t;
      })
      $(".ol").remove();
      $("#offloadops").html(`Following ops were offloaded to <span id='nnbackend' class='ols'></span> from <span id='polyfillbackend' class='ols'></span>: `);
      $("#offloadops").append(offloadopsvalue).append(`<span data-toggle="modal" class="subgraph-btn" data-target="#subgraphModal">View Subgraphs</span>`);
      $("#nnbackend").html(prefer);
      $("#polyfillbackend").html(backend);
    } else {
      $('.offload').hide();
    }
  };

  let intersection = new Set(supportedOps.filter(x => requiredOps.has(x)));
  console.log('NN supported: ' + supportedOps);
  console.log('Model required: ' + [...requiredOps]);
  console.log('Ops offload: ' + [...intersection]);
  hybridRow(intersection, backend, prefer);
};

const showSubGraphsSummary = (summary, multi = false) => {
  if (summary) {
    let listhtml = '';
    for (let i in summary) {
      let backend = summary[i].split(':')[0].toLowerCase();
      let subgraphlist = summary[i].split(':')[1].replace(/ /g, '').replace('{', '').replace('}', '').replace(/,/g, ' ');
      let tmp;
      if (backend.indexOf('webnn') > -1) {
        tmp = `<li><div class="timeline-badge tb-webnn"><i class="glyphicon">WebNN</i></div><div class="timeline-panel tp-webnn"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      } else if (backend.indexOf('wasm') > -1) {
        tmp = `<li class="timeline-inverted"><div class="timeline-badge tb-wasm"><i class="glyphicon">WASM</i></div><div class="timeline-panel tp-wasm"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      } else if (backend.indexOf('webgl') > -1) {
        tmp = `<li class="timeline-inverted"><div class="timeline-badge tb-webgl"><i class="glyphicon">WebGL</i></div><div class="timeline-panel tp-webgl"><div class="timeline-body"><p>${subgraphlist}</p></div></div></li>`;
      }
      listhtml += tmp;
    }
    $('#subgraph').html(listhtml);
  }
};

const componentToggle = () => {
  $('#header-sticky-wrapper').slideToggle();
  $('#query').slideToggle();
  $('.nav-pills').slideToggle();
  $('.github-corner').slideToggle();
  $('footer').slideToggle();
  $('#extra span').toggle();
};
