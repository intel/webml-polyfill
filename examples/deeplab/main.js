const availableModels = [
  {
    modelName: 'DeepLab 513',
    modelFile: './model/deeplab_mobilenetv2_513.tflite',
    labelsFile: './model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 21],
  },
  {
    modelName: 'DeepLab 513 Atrous',
    modelFile: './model/deeplab_mobilenetv2_513_dilated.tflite',
    labelsFile: './model/labels.txt',
    inputSize: [513, 513, 3],
    outputSize: [513, 513, 21],
  },
];

function main(camera) {

  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const selectPrefer = document.getElementById('selectPrefer');
  const backend = document.getElementById('backend');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const zoomSlider = document.getElementById('zoomSlider');
  const blurSlider = document.getElementById('blurSlider');
  const refineEdgeSlider = document.getElementById('refineEdgeSlider');
  const colorMapAlphaSlider = document.getElementById('colorMapAlphaSlider');
  const selectBackgroundButton = document.getElementById('chooseBackground');
  const clearBackgroundButton = document.getElementById('clearBackground');
  const outputCanvas = document.getElementById('output');
  let clippedSize = [];
  let currentBackend = '';
  let currentModel = '';
  let currentPrefer = '';
  let streaming = false;
  let hoverPos = null;
  let stats;

  const counterN = 20;
  let counter = 0;
  let inferTimeAcc = 0;
  let drawTimeAcc = 0;


  let renderer = new Renderer(outputCanvas);
  renderer.setup();

  let utils = new Utils();
  // register updateProgress function if progressBar element exist
  utils.progressCallback = updateProgress;

  let colorPicker = new iro.ColorPicker("#color-picker-container", {
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
  colorPicker.on('color:change', function(color) {
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



  $('.effects-select .btn input').filter(function() {
    return this.value === renderer.effect;
  }).parent().toggleClass('active');
  $('.controls').attr('data-select', renderer.effect);
  $('.effects-select .btn').click((e) => {
    e.preventDefault();
    let effect = e.target.children[0].value;
    $('.controls').attr('data-select', effect);
    renderer.effect = effect;
  });


  function checkPreferParam() {
    if (currentOS === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Currently ${backend} backend doesn't support DeepLab Model.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function showPreferAlert() {
    let div = document.createElement('div');
    div.setAttribute('id', 'preferAlert');
    div.setAttribute('class', 'alert alert-danger alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Invalid prefer, prefer should be 'fast' or 'sustained'.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function removeAlertElement() {
    let backendAlertElem = document.getElementById('backendAlert');
    if (backendAlertElem !== null) {
      backendAlertElem.remove();
    }
    let preferAlertElem = document.getElementById('preferAlert');
    if (preferAlertElem !== null) {
      preferAlertElem.remove();
    }
  }

  function updateBackend() {
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI(currentPrefer) : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    streaming = false;
    if (newBackend !== "WebML") {
      selectPrefer.style.display = 'none';
    } else {
      selectPrefer.style.display = 'inline';
    }
    // renderer.deleteAll();
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend, currentPrefer).then(() => {
        currentBackend = newBackend;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function updateModel() {
    selectModel.innerHTML = currentModel;
  }

  function changeModel(newModel) {
    if (currentModel === newModel.modelName) {
      return;
    }
    streaming = false;
    // renderer.deleteAll();
    utils.deleteAll();
    removeAlertElement();
    utils.changeModelParam(newModel);
    currentPrefer = "sustained";
    progressContainer.style.display = "inline";
    selectModel.innerHTML = 'Setting...';

    setTimeout(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        currentModel = newModel.modelName;
        updatePrefer();
        updateBackend();
        updateModel();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          let res = utils.getFittedResolution(4 / 3);
          setCamResolution(res).then(() => {
            streaming = true;
            startPredict();
          });
        }
      });
    }, 10);
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    // renderer.deleteAll();
    utils.deleteAll();
    removeAlertElement();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, newPrefer).then(() => {
        currentPrefer = newPrefer;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          predictAndDraw(imageElement);
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        let currentBackend = 'WebML/' + getNativeAPI(currentPrefer);
        let nextBackend = 'WebML/' + getNativeAPI(newPrefer);
        console.warn(`Failed to change backend ${nextBackend}, switch back to ${currentBackend}`);
        console.error(e);
        changePrefer(currentPrefer, true);
        showAlert(nextBackend);
        updatePrefer();
        updateBackend();
      });
    }, 10);
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
  }

  function updateProgress(ev) {
    if (ev.lengthComputable) {
      let percentComplete = ev.loaded / ev.total * 100;
      percentComplete = percentComplete.toFixed(0);
      progressBar.style = `width: ${percentComplete}%`;
      progressBar.innerHTML = `${percentComplete}%`;
      if (ev.loaded === ev.total) {
        progressContainer.style.display = "none";
        progressBar.style = `width: 0%`;
        progressBar.innerHTML = `0%`;
      }
    }
  }

  async function predictAndDraw(imageSource) {
    clippedSize = utils.prepareInput(imageSource);
    renderer.uploadNewTexture(imageSource, clippedSize);
    let result = await utils.predict();
    let inferTime = result.time;
    console.log(`Inference time: ${inferTime.toFixed(2)} ms`);
    inferenceTime.innerHTML = `inference time: <em style="color:green;font-weight:bloder">${inferTime.toFixed(2)} </em>ms`;
    renderer.drawOutputs(result.segMap)
      // .then((drawTime) => {
      //   inferTimeAcc += inferTime;
      //   drawTimeAcc += drawTime;
      //   if (++counter === counterN) {
      //     console.debug(`(${counterN} frames) Infer time: ${(inferTimeAcc / counterN).toFixed(2)} ms`);
      //     console.debug(`(${counterN} frames) Draw time: ${(drawTimeAcc / counterN).toFixed(2)} ms`);
      //     counter = inferTimeAcc = drawTimeAcc = 0;
      //   }
      // });
    renderer.highlightHoverLabel(hoverPos);
  }


  // register backends
  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    };
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function (e) {
      removeAlertElement();
      changeBackend('WebGL');
    };
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function (e) {
      removeAlertElement();
      changeBackend('WASM');
    };
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WebGL';
    }
  }

  function setCamResolution(resolution) {
    return navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user' }
    }).then((stream) => {
      videoElement.srcObject = stream;
      return new Promise((resolve) => {
        // video cannot be uploaded to texture until being loaded
        videoElement.onloadeddata = resolve;
      });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });
  }

  // register models
  for (let model of availableModels) {
    let dropdownBtn = $('<button class="dropdown-item d-flex"/>')
      .append(
        $('<div class="model-link"/>')
          .text(model.modelName)
          .click(_ => changeModel(model))
      ).append(
        $('<div class="netron-link ml-auto pl-2">')
          .text('▶')
          .click(_ => {
            let modelUrl = new URL(model.modelFile, window.location.href).href;
            window.open(`https://lutzroeder.github.io/netron/?url=${modelUrl}`);
          })
      );

    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      currentModel = model.modelName;
    }
  }

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

  // register prefers
  if (currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let sustained = $('<button class="dropdown-item"/>')
      .text('SUSTAINED_SPEED')
      .click(_ => changePrefer('sustained'));
    $('.preference').append(sustained);
    if (currentOS === 'Android') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item"/>')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Windows' || currentOS === 'Linux') {
      let fast = $('<button class="dropdown-item" disabled />')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    } else if (currentOS === 'Mac OS') {
      let fast = $('<button class="dropdown-item"/>')
        .text('FAST_SINGLE_ANSWER')
        .click(_ => changePrefer('fast'));
      $('.preference').append(fast);
      let low = $('<button class="dropdown-item" disabled />')
        .text('LOW_POWER')
        .click(_ => changePrefer('low'));
      $('.preference').append(low);
    }
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

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

  // image or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
      $('.credit').remove();
    }, false);
    let imageWrapper = document.getElementsByClassName('image-wrapper')[0];
    imageWrapper.ondragover = (e) => {
      e.preventDefault();
    };
    imageWrapper.ondragenter = (e) => {
      e.preventDefault();
      $('.image-wrapper').addClass('show');
    };
    imageWrapper.ondragleave = (e) => {
      e.preventDefault();
      $('.image-wrapper').removeClass('show');
    };
    imageWrapper.ondrop = (e) => {
      e.preventDefault();
      $('.image-wrapper').removeClass('show');
      let files = e.dataTransfer.files;
      if (files.length > 0 && files[0].type.split('/')[0] === 'image') {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    };

    imageElement.onload = function () {
      predictAndDraw(imageElement);
    };

    utils.init(currentBackend, currentPrefer).then(() => {
      updateBackend();
      updateModel();
      updatePrefer();
      predictAndDraw(imageElement);
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    }).catch((e) => {
      console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
      console.error(e);
      showAlert(utils.model._backend);
      changeBackend('WASM');
    });
  } else {
    stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    let res = utils.getFittedResolution(4 / 3);
    setCamResolution(res).then(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        updateBackend();
        updateModel();
        updatePrefer();
        streaming = true;
        startPredict();
      }).catch((e) => {
        console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
        console.error(e);
        showAlert(utils.model._backend);
        changeBackend('WASM');
      });
    })
  }


  function startPredict() {
    if (streaming) {
      stats.begin();
      predictAndDraw(videoElement).then(_ => {
        stats.end();
        setTimeout(startPredict, 0);
      });
    }
  }
}
