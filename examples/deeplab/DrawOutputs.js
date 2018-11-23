const palette = [
  [45, 52, 54, 255],
  [85, 239, 196, 255],
  [129, 236, 236, 255],
  [116, 185, 255, 255],
  [162, 155, 254, 255],
  [223, 230, 233, 255],
  [0, 184, 148, 255],
  [0, 206, 201, 255],
  [9, 132, 227, 255],
  [39, 60, 117, 255],
  [108, 92, 231, 255],
  [178, 190, 195, 255],
  [255, 234, 167, 255],
  [250, 177, 160, 255],
  [255, 118, 117, 255],
  [253, 121, 168, 255],
  [99, 110, 114, 255],
  [253, 203, 110, 255],
  [225, 112, 85, 255],
  [214, 48, 49, 255],
  [232, 67, 147, 255],
];

function drawSegMap(canvas, segMap, highlightId) {
  let recover;
  if (typeof highlightId !== 'undefined') {
    let highlight = palette[highlightId];
    recover = highlight.slice(0); // clone
    highlight[0] = Math.max(highlight[0] - 30, 0);
    highlight[1] = Math.max(highlight[1] - 30, 0);
    highlight[2] = Math.max(highlight[2] - 30, 0);
  }

  _drawSegMap(canvas, segMap, highlightId);

  if (typeof highlightId !== 'undefined')
    palette[highlightId] = recover;
}

function _drawSegMap(canvas, segMap) {
  // colorize output seg map
  const outputWidth = 65;
  const outputHeight = 65;
  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = outputWidth;
  outputCanvas.height = outputHeight;
  const outputCtx = outputCanvas.getContext('2d');
  const imgData = outputCtx.getImageData(0, 0, outputWidth, outputHeight);
  const colorSegMap = segMap.data.flatMap(c => palette[c]);
  imgData.data.set(new Uint8ClampedArray(colorSegMap));
  outputCtx.putImageData(imgData, 0, 0);

  // scale and draw
  const scaleWidth = segMap.shape[0];
  const scaleHeight = segMap.shape[1];
  canvas.width = scaleWidth;
  canvas.height = scaleHeight;
  const ctx = canvas.getContext('2d');
  const scaleSize = Math.max(scaleWidth, scaleWidth);
  ctx.drawImage(outputCanvas, 0, 0, scaleSize, scaleSize);
}