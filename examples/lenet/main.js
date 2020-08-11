const predictButton = document.getElementById('predict');
const nextButton = document.getElementById('next');
const resultDiv = document.getElementById('result');
const visualContext = document.getElementById('visual_canvas').getContext('2d');
const digitCanvas = document.createElement('canvas');
const height = 28;
const width = 28;
digitCanvas.setAttribute('height', height);
digitCanvas.setAttribute('width', width);
const digitContext = digitCanvas.getContext('2d');
let digit;

function generateRandomDigit() {
  const n = Math.floor(Math.random() * 10);
  digit = mnist[n].get();
  mnist.draw(digit, digitContext);
  visualContext.drawImage(digitCanvas, 0, 0, 280, 280);
}

async function main() {
  generateRandomDigit();
  const lenet = new Lenet('lenet.bin');
  try {
    let start = performance.now();
    await lenet.load();
    console.log(`loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    await lenet.compile();
    console.log(`compilation elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    predictButton.removeAttribute('disabled');
  } catch (error) {
    addWarning(error.message);
  }
  predictButton.addEventListener('click', async function (e) {
    try {
      let start = performance.now();
      const result = await lenet.predict(digit);
      console.log(`execution elapsed time: ${(performance.now() - start).toFixed(2)} ms`);
      console.log(`execution result: ${result}`);
      let resultContent = '';
      const classes = topK(result);
      for (c of classes) {
        resultContent += `${c.label}: ${c.prob}\%<br>`
      }
      resultDiv.innerHTML = resultContent;
    } catch (error) {
      addWarning(error.message);
    }
  });
  nextButton.addEventListener('click', () => {
    generateRandomDigit();
    resultDiv.innerHTML = '';
  });
}

function topK(probs, k = 3) {
  const sorted = probs.map((prob, index) => [prob, index]).sort((a, b) => {
    if (a[0] === b[0]) {
      return 0;
    }
    return a[0] < b[0] ? -1 : 1;
  });
  sorted.reverse();

  const classes = [];
  for (let i = 0; i < k; ++i) {
    let c = {
      label: sorted[i][1],
      prob: (sorted[i][0] * 100).toFixed(2)
    }
    classes.push(c);
  }

  return classes;
}

function addWarning(msg) {
  let div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = msg;
  let container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

function removeWarning() {
  $('.alert').alert('close')
}
