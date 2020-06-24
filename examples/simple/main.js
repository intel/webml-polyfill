const computeButton = document.getElementById('compute')
const inputElement1 = document.getElementById('input1');
const inputElement2 = document.getElementById('input2');
const resultElement = document.getElementById('result');

async function main() { 
    const simpleModel = new SimpleModel('model_data.bin');

    let start = performance.now();
    await simpleModel.load();
    console.log(`loading elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    start = performance.now();
    await simpleModel.compile({ powerPreference: 'low-power' });
    console.log(`compilation elapsed time: ${(performance.now() - start).toFixed(2)} ms`);

    computeButton.removeAttribute('disabled');
    computeButton.addEventListener('click', async function (e) {
      const input1 = parseFloat(inputElement1.value);
      const input2 = parseFloat(inputElement2.value);
      if (isNaN(input1) || isNaN(input2)) {
        console.log('Invalid inputs');
        resultElement.innerHTML = '';
        addWarning();
        return;
      } else {
        removeWarning();
      }
      start = performance.now();
      const result = await simpleModel.compute(input1, input2);
      console.log(`execution elapsed time: ${(performance.now() - start).toFixed(2)} ms`);
      console.log(`execution result: ${result}`);
      resultElement.innerHTML = result;
    });
}


function addWarning() {
  let div = document.createElement('div');
  div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
  div.setAttribute('role', 'alert');
  div.innerHTML = '<strong>Invalid inputs!</strong> Please input valid float numbers in below fields.';
  let container = document.getElementById('container');
  container.insertBefore(div, container.childNodes[0]);
}

function removeWarning() {
  $('.alert').alert('close')
}
