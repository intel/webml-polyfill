const buttonUI = (camera = false) => {
  if (camera) {
    $('#pickimage').hide();
    $('#fps').show();
  } else {
    $('#pickimage').show();
    $('#fps').hide();
  }
}

const setFullScreenIconPosition = (modelname) => {
  let svgstyle = 'p' + modelname.replace('deeplab_mobilenet_v2_', '').replace('_tflite', '').replace(/_/g, '').replace('atrous', '');
  // $('#semanticsegmentation #fullscreen i svg').removeClass('p224').removeClass('p257').removeClass('p321').removeClass('p513').addClass(svgstyle);
  $('#semanticsegmentation #fullscreen i svg').addClass('p513');
}

let ssmodel = () => {
  return um.replace('mobilenet', '').replace('v2', '').replace(/_/g, ' ');
}

$(document).ready(() => {

  if (hasUrlParam('m') && hasUrlParam('t')) {
    setFullScreenIconPosition(um);
  }

  updateTitle('Semantic Segmentation', ub, up, ssmodel(), ut);

  $('input:radio[name=bp]').click(() => {
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    let rid = $('input:radio[name="m"]:checked').attr('id');
    if(rid) {
      setFullScreenIconPosition(rid);
    }
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
  });
 
});

$(document).ready(() => {
  $('#img').click(() => {
    buttonUI(us === 'camera');
  });

  $('#cam').click(() => {
    buttonUI(us === 'camera');
  });

  $('#fullscreen i svg').click(() => {
    $('#canvasvideo').toggleClass('fullscreen');
    $('.zoom-wrapper').toggle();
    $('#labelitem').toggle();
  });

});

const showResultsSS = () => {
  $('#progressmodel').hide();
  $('.icdisplay').show();
  $('.shoulddisplay').show();
  $('#resulterror').hide();
  buttonUI(us === 'camera');
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

  const doubleZoomLevel = (modelname) => {
    let doublezoomlevel = modelname.replace('deeplab_mobilenet_v2_', '').replace('_tflite', '').replace(/_/g, '').replace('atrous', '');
    if (doublezoomlevel) {
      switch (parseInt(doublezoomlevel)) {
        case 513:
          renderer.zoom = 1;
          zoomSlider.value = 100;
          break;
        case 224:
          renderer.zoom = 2.3;
          zoomSlider.value = 2.3;
          break;
        case 257:
          renderer.zoom = 2;
          zoomSlider.value = 2;
          break;
        case 321:
          renderer.zoom = 1.6;
          zoomSlider.value = 1.6;
          break;
        default:
          renderer.zoom = 1;
          zoomSlider.value = 100;
      }
    }
  }

  doubleZoomLevel(um);
  $('.zoom-value').html(renderer.zoom + 'x');
  zoomSlider.oninput = () => {
    let zoom = zoomSlider.value / 100;
    $('.zoom-value').html(zoom + 'x');
    renderer.zoom = zoom;
  };

  $('input:radio[name=m]').click(() => {
    let rid = $('input:radio[name="m"]:checked').attr('id');
    doubleZoomLevel(rid);
  });

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
  if (um === 'none') {
    showError('No model selected', 'Please select a model to start prediction.');
    return;
  }
  main(us === 'camera');
})