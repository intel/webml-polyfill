class SemanticSegmentationExample extends BaseCameraExample {
  constructor(models) {
    super(models);
    this._renderer = new Renderer(document.getElementById('canvasvideo'));
    this._hoverPos = null;
  }

  _setHoverPos = (p) => {
    this._hoverPos = p;
  };

  _customUI = () => {
    $('#fullscreen i svg').click(() => {
      $('#canvasvideo').toggleClass('fullscreen');
      $('.zoom-wrapper').toggle();
    });

    $('#cam').click(() => {
      $('#pickimage').hide();
    });

    this._renderer.setup();
  };

  loadedUI = () => {
    let _this =  this;
    const blurSlider = document.getElementById('blurSlider');
    const refineEdgeSlider = document.getElementById('refineEdgeSlider');
    const colorMapAlphaSlider = document.getElementById('colorMapAlphaSlider');
    const selectBackgroundButton = document.getElementById('chooseBackground');
    const clearBackgroundButton = document.getElementById('clearBackground');
    const outputCanvas = document.getElementById('canvasvideo');
    let colorPicker = new iro.ColorPicker('#color-picker-container', {
      width: 200,
      height: 200,
      color: {
        r: _this._renderer.bgColor[0],
        g: _this._renderer.bgColor[1],
        b: _this._renderer.bgColor[2]
      },
      markerRadius: 5,
      sliderMargin: 12,
      sliderHeight: 20,
    });

    $('.bg-value').html(colorPicker.color.hexString);

    colorPicker.on('color:change', (color) => {
      $('.bg-value').html(color.hexString);
      _this._renderer.bgColor = [color.rgb.r, color.rgb.g, color.rgb.b];
    });

    $('input:radio[name=m]').click(() => {
      let rid = $('input:radio[name="m"]:checked').attr('id');
    });

    colorMapAlphaSlider.value = _this._renderer.colorMapAlpha * 100;
    $('.color-map-alpha-value').html(_this._renderer.colorMapAlpha);

    colorMapAlphaSlider.oninput = () => {
      let alpha = colorMapAlphaSlider.value / 100;
      $('.color-map-alpha-value').html(alpha);
      _this._renderer.colorMapAlpha = alpha;
    };

    blurSlider.value = _this._renderer.blurRadius;
    $('.blur-radius-value').html(_this._renderer.blurRadius + 'px');

    blurSlider.oninput = () => {
      let blurRadius = parseInt(blurSlider.value);
      $('.blur-radius-value').html(blurRadius + 'px');
      _this._renderer.blurRadius = blurRadius;
    };

    refineEdgeSlider.value = _this._renderer.refineEdgeRadius;

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
      _this._renderer.refineEdgeRadius = refineEdgeRadius;
    };

    $('.effects-select .btn input').filter(() => {
      return this.value === _this._renderer.effect;
    }).parent().toggleClass('active');

    $('.controls').attr('data-select', _this._renderer.effect);

    $('.effects-select .btn').click((e) => {
      e.preventDefault();
      let effect = e.target.children[0].value;
      $('.controls').attr('data-select', effect);
      _this._renderer.effect = effect;
    });

    selectBackgroundButton.addEventListener('change', (e) => {
      const files = e.target.files;
      if (files.length > 0) {
        const img = new Image();
        img.onload = () => {
          _this._renderer.backgroundImageSource = img;
        };
        img.src = URL.createObjectURL(files[0]);
      }
    }, false);

    clearBackgroundButton.addEventListener('click', (e) => {
      _this._renderer.backgroundImageSource = null;
    }, false);

    outputCanvas.addEventListener('mousemove', (e) => {
      const getMousePos = (canvas, evt) => {
        let rect = canvas.getBoundingClientRect();
        return {
          x: Math.ceil(evt.clientX - rect.left),
          y: Math.ceil(evt.clientY - rect.top)
        };
      };

      _this._setHoverPos(getMousePos(outputCanvas, e));
      _this._renderer.highlightHoverLabel(_this._hoverPos);
    });

    outputCanvas.addEventListener('mouseleave', (e) => {
      _this._setHoverPos(null);
      _this._renderer.highlightHoverLabel(_this._hoverPos);
    });
  };

  _createRunner = () => {
    const runner = new SemanticSegmentationRunner();
    runner.setProgressHandler(updateLoadingProgressComponent);
    return runner;
  };

  _predict = async () => {
    const drawOptions = {
      inputSize: this._currentModelInfo.inputSize,
      preOptions: this._currentModelInfo.preOptions,
      imageChannels: 4,
      scaledFlag: true,
    };
    await this._runner.run(this._currentInputElement, drawOptions);
    this._processOutput();
  };

  _processCustomOutput = () => {
    const width = this._currentModelInfo.inputSize[1];
    const imWidth = this._currentInputElement.naturalWidth | this._currentInputElement.videoWidth;
    const imHeight = this._currentInputElement.naturalHeight | this._currentInputElement.videoHeight;
    const resizeRatio = Math.max(Math.max(imWidth, imHeight) / width, 1);
    const scaledWidth = Math.floor(imWidth / resizeRatio);
    const scaledHeight = Math.floor(imHeight / resizeRatio);
    const output = this._runner.getOutput();
    const segMap = {
      data: output.outputTensor,
      outputShape: this._currentModelInfo.outputSize,
      labels: output.labels,
    };
    this._renderer.uploadNewTexture(this._currentInputElement, [scaledWidth, scaledHeight]);
    this._renderer.drawOutputs(segMap);
    this._renderer.highlightHoverLabel(this._hoverPos);

    if (this._currentInputType === 'image') {
      $('#pickimage').show();
    } else {
      $('#pickimage').hide();
    }
  };
}