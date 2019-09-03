// used for face detection
function drawFaceBoxes(image, canvas, face_boxes, classes) {
  canvas.height = 300;
  canvas.width = image.width / image.height * canvas.height;
  // drawImage
  ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  // drawFaceBox
  face_boxes.forEach((box, i) => {
    let xmin = Math.max(box[0] / image.height * canvas.height, 0);
    let xmax = Math.min(box[1] / image.height * canvas.height, canvas.width);
    let ymin = Math.max(box[2] / image.height * canvas.height, 0);
    let ymax = Math.min(box[3] / image.height * canvas.height, canvas.height);
    ctx.strokeStyle = "#009bea";
    ctx.fillStyle = "#009bea";
    ctx.lineWidth = 3;
    ctx.strokeRect(xmin, ymin, xmax-xmin, ymax-ymin);
    ctx.font = "20px Arial";
    let text = classes[i];
    let width = ctx.measureText(text).width;
    if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
      ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
      ctx.fillStyle = "white";
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin, ymin - 3);
    } else {
      ctx.fillRect(xmin + 2, ymin , width + 4,  parseInt(ctx.font, 10));
      ctx.fillStyle = "white";
      ctx.textAlign = 'start';
      ctx.fillText(text, xmin + 2, ymin + 15);
    }
  });
}
