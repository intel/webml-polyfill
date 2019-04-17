// used for face detection
function drawFaceBoxes(image, canvas, face_boxes) {
    canvas.width = image.width / image.height * canvas.height;
    // drawImage
    ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  
    // drawFaceBox
    face_boxes.forEach(box => {
      let xmin = box[0] / image.height * canvas.height;
      let xmax = box[1] / image.height * canvas.height;
      let ymin = box[2] / image.height * canvas.height;
      let ymax = box[3] / image.height * canvas.height;
      let prob = box[4];
      ctx.strokeStyle = "#009bea";
      ctx.fillStyle = "#009bea";
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax-xmin, ymax-ymin);
      ctx.font = "18px Arial";
      let text = `${prob.toFixed(2)}`;
      let width = ctx.measureText(text).width;
      if (xmin >= 2 && ymin >= parseInt(ctx.font, 10)) {
        ctx.fillRect(xmin - 2, ymin - parseInt(ctx.font, 10), width + 4, parseInt(ctx.font, 10));
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin, ymin - 3);
      } else {
        ctx.fillRect(xmin + 2, ymin , width + 4,  parseInt(ctx.font, 10));
        ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
        ctx.textAlign = 'start';
        ctx.fillText(text, xmin + 2, ymin + 15);
      }
    });
  }
  
// used for face landmark detection
function drawKeyPoints(image, canvas, Keypoints, boxes) {
  ctx = canvas.getContext('2d');
  boxes.forEach((box, n) => {
    keypoints = Keypoints[n];
    for (let i = 0; i < 136; i = i + 2) {
      // decode keypoints
      let x = ((box[1] - box[0]) * keypoints[i] + box[0]) / image.height * canvas.height;
      let y = ((box[3] - box[2]) * keypoints[i + 1] + box[2]) / image.height * canvas.height;
      // draw keypoints
      ctx.beginPath();
      ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.closePath();
    }
  });
}