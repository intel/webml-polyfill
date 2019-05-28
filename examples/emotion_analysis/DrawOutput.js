// used for face detection
function drawFaceBoxes(image, canvas, face_boxes, classes) {
    canvas.width = image.width / image.height * canvas.height;
    // drawImage
    ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  
    // drawFaceBox
    face_boxes.forEach((box, i) => {
      let xmin = box[0] / image.height * canvas.height;
      let xmax = box[1] / image.height * canvas.height;
      let ymin = box[2] / image.height * canvas.height;
      let ymax = box[3] / image.height * canvas.height;
      ctx.strokeStyle = "#009bea";
      ctx.fillStyle = "#009bea";
      ctx.lineWidth = 3;
      ctx.strokeRect(xmin, ymin, xmax-xmin, ymax-ymin);
      ctx.font = "20px Arial";
      let prob = classes[i].prob;
      let label = classes[i].label;
      let text = `${label}:${prob}`;
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

function getTopClasses(tensors, labels, k = 3) {
  let classes = [];
  tensors.forEach(tensor => {
    let probs = Array.from(tensor);
    let indexes = probs.map((prob, index) => [prob, index]);
    let sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) {return 0;}
      return a[0] < b[0] ? -1 : 1;
    });
    sorted.reverse();
    for (let i = 0; i < k; ++i) {
      let prob = sorted[i][0];
      let index = sorted[i][1];
      let c = {
        label: labels[index],
        prob: (prob * 100).toFixed(2)
      }
      classes.push(c);
    }
  });
  return classes;
}