let colorizer = new Worker('SegMapColorizer.js');

function drawSegMap(canvas, segMap) {
  colorizer.postMessage(segMap);

  let outputWidth = segMap.outputShape[0];
  let outputHeight = segMap.outputShape[1];
  let scaledWidth = segMap.scaledShape[0];
  let scaledHeight = segMap.scaledShape[1];
  let ctx = canvas.getContext('2d');
  let imgData = ctx.getImageData(0, 0, outputWidth, outputHeight);

  colorizer.onmessage = function(e) {
    let colorSegMap = e.data[0];
    let labelMap = e.data[1];
    imgData.data.set(colorSegMap);
    canvas.width = scaledWidth;
    canvas.height = scaledHeight;
    ctx.putImageData(imgData, 0, 0, 0, 0, scaledWidth, scaledHeight);
    showLegends(labelMap);
  };
}

function showLegends(labelMap) {
  $('.labels-wrapper').empty();
  for (let labelId in labelMap) {
    let labelDiv =
      $(`<div class="col-12 seg-label" data-label-id="${labelId}"/>`)
      .append($(`<span style="color: rgb(${labelMap[labelId][1]})">⬤</span>`))
      .append(`${labelMap[labelId][0]}`);
    // labelDiv.mouseenter(_ => drawSegMap(segMapCanvas, segMap, labelId));
    // labelDiv.mouseleave(_ => drawSegMap(segMapCanvas, segMap));
    $('.labels-wrapper').append(labelDiv);
  }
}