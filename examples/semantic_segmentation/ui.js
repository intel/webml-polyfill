const buttonUI = (camera = false) => {
  if (camera) {
    $('#pickimage').hide();
    $('#fps').show();
  } else {
    $('#pickimage').show();
    $('#fps').hide();
  }
}

let ssmodel = () => {
  return um.replace('mobilenet', '').replace('v2', '').replace(/_/g, ' ');
}

$(document).ready(() => {
  updateTitle('Semantic Segmentation', ub, up, ssmodel(), ut);

  $('input:radio[name=bp]').click(() => {
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
  });

  $('input:radio[name=bw]').click(() => {
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
  });

  $('input:radio[name=m]').click(() => {
    updateTitle('Semantic Segmentation', currentBackend, currentPrefer, ssmodel(), `${ut}`);
    $('.offload').hide();
  });
 
  inputElement.addEventListener('change', (e) => {
    let files = e.target.files;
    if (files.length > 0) {
      imageElement.src = URL.createObjectURL(files[0]);
    }
  }, false);

  imageElement.addEventListener('load', () => {
    predictAndDraw(imageElement, false);
  }, false);
 
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
  });

});

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

  $('input:radio[name=m]').click(() => {
    let rid = $('input:radio[name="m"]:checked').attr('id');
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