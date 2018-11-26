class MultiTaskWorker extends Worker {
  constructor(src) {
    super(src);
    this._handlers = {};
    this.onmessage = function (e) {
      let fnId = e.data[1] + e.data[0];
      let ret = e.data[2];
      this._handlers[fnId](ret);
      delete this._handlers[fnId];
    };
    this.dispatch = function (fn, msg) {
      return new Promise((resolve) => {
        let id = Date.now().toString();
        this.postMessage([id, fn, msg.args], msg.transferList);
        this._handlers[fn + id] = resolve;
      });
    };
  }
}

const pp = new MultiTaskWorker('PostProcessor.js');

function drawSegMap(canvas, segMap) {
  let outputWidth = segMap.outputShape[0];
  let outputHeight = segMap.outputShape[1];
  let scaledWidth = segMap.scaledShape[0];
  let scaledHeight = segMap.scaledShape[1];

  pp.dispatch('colorizeAndPredictLabels', {
      args: [segMap],
      transferList: [segMap.data.buffer],
    })
    .then(([colorSegMap, labelMap]) => {
      let ctx = canvas.getContext('2d');
      let imgData = ctx.getImageData(0, 0, outputWidth, outputHeight);
      imgData.data.set(colorSegMap);
      canvas.width = scaledWidth;
      canvas.height = scaledHeight;
      ctx.putImageData(imgData, 0, 0, 0, 0, scaledWidth, scaledHeight);
      showLegends(labelMap);
    });
}

function highlightHoverLabel(hoverPos) {
  if (hoverPos === null) {
    // clear highlight when mouse leaving canvas
    $('.seg-label').removeClass('highlight');
    return;
  }

  pp.dispatch('getHoverLabelId', {
      args: [hoverPos],
    })
    .then(([id]) => {
      $('.seg-label').removeClass('highlight');
      $('.labels-wrapper').find(`[data-label-id="${id}"]`).addClass('highlight');
    });
}

function showLegends(labelMap) {
  $('.labels-wrapper').empty();
  for (let id in labelMap) {
    let labelDiv = $(`<div class="col-12 seg-label" data-label-id="${id}"/>`)
      .append($(`<span style="color:rgb(${labelMap[id][1]})">⬤</span>`))
      .append(`${labelMap[id][0]}`);
    $('.labels-wrapper').append(labelDiv);
  }
}