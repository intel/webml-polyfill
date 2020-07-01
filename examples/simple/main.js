const computeButton = document.getElementById('compute');
const constant1Element = document.getElementById('constant1');
const constant2Element = document.getElementById('constant2');
const input1Element = document.getElementById('input1');
const input2Element = document.getElementById('input2');
const resultElement = document.getElementById('result');

async function main() {
  const simpleModel = new SimpleModel('model_data.bin');
  try {
    let start = performance.now();
    const [constant1Value, constant2Value] = await simpleModel.load();
    console.log(`loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);
    constant1Element.innerHTML = constant1Value;
    constant2Element.innerHTML = constant2Value;

    start = performance.now();
    await simpleModel.compile({ powerPreference: 'low-power' });
    console.log(`compilation elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    computeButton.removeAttribute('disabled');
  } catch (error) {
    addWarning(error.message);
  }
  computeButton.addEventListener('click', async function (e) {
    const input1 = parseFloat(input1Element.value);
    const input2 = parseFloat(input2Element.value);
    if (isNaN(input1) || isNaN(input2)) {
      console.log('Invalid inputs');
      resultElement.innerHTML = '';
      addWarning('<strong>Invalid inputs!</strong> Please input valid float numbers in below fields.');
      return;
    } else {
      removeWarning();
    }
    try {
      let start = performance.now();
      const result = await simpleModel.compute(input1, input2);
      console.log(`execution elapsed time: ${(performance.now() - start).toFixed(2)} ms`);
      console.log(`execution result: ${result}`);
      resultElement.innerHTML = result;
    } catch (error) {
      addWarning(error.message);
    }
  });
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
