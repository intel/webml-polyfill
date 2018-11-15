const _palette = [
  [85, 239, 196],
  [129, 236, 236],
  [116, 185, 255],
  [162, 155, 254],
  [223, 230, 233],
  [0, 184, 148],
  [0, 206, 201],
  [9, 132, 227],
  [39, 60, 117],
  [108, 92, 231],
  [178, 190, 195],
  [255, 234, 167],
  [250, 177, 160],
  [255, 118, 117],
  [253, 121, 168],
  [99, 110, 114],
  [253, 203, 110],
  [225, 112, 85],
  [214, 48, 49],
  [232, 67, 147],
  [45, 52, 54],
];

function drawSegMap(canvas, segMap, alpha = 0.8) {
  const data = segMap.data;
  const scaleWidth = segMap.shape[0];
  const scaleHeight = segMap.shape[1];
  const imageChannels = 4; // RGBA
  canvas.width = scaleWidth;
  canvas.height = scaleHeight;

  const outputWidth = 65;
  const outputHeight = 65;
  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = outputWidth;
  outputCanvas.height = outputHeight;
  const outputCtx = outputCanvas.getContext('2d');
  const imgData = outputCtx.createImageData(outputWidth, outputHeight);
  for (let i = 0; i < data.length; i++) {
    let color = _palette[data[i]];
    imgData.data[i*imageChannels+0] = color[0];
    imgData.data[i*imageChannels+1] = color[1];
    imgData.data[i*imageChannels+2] = color[2];
    imgData.data[i*imageChannels+3] = alpha * 255;
  }

  const ctx = canvas.getContext('2d');
  let destImgData = ctx.createImageData(scaleWidth, scaleHeight);
  bilinear(imgData, destImgData, 513 / 65);
  ctx.putImageData(destImgData, 0, 0);
}