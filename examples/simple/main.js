const computeButton = document.getElementById('compute')
const inputElement1 = document.getElementById('input1');
const inputElement2 = document.getElementById('input2');
const resultElement = document.getElementById('result');

async function loadModelDataFile(url) {
  let response = await fetch(url);
  let bytes = await response.arrayBuffer();
  return bytes;
}

function main() {
  loadModelDataFile('model_data.bin').then(bytes => {
    let simpleModel = new SimpleModel(bytes);
    simpleModel.createCompiledModel().then(result => {
      console.log(`compilation result: ${result}`);
      computeButton.removeAttribute('disabled');
      computeButton.addEventListener('click', e => {
        let input1 = parseFloat(inputElement1.value);
        let input2 = parseFloat(inputElement2.value);
        if (isNaN(input1) || isNaN(input2)) {
          console.log('Invalid inputs');
          resultElement.innerHTML = '';
          addWarning();
          return;
        } else {
          removeWarning();
        }
        let start = performance.now();
        simpleModel.compute(input1, input2).then(result => {
          let elapsed = performance.now() - start;
          console.log(`execution elapsed time: ${elapsed.toFixed(2)} ms`);
          console.log(`execution result: ${result}`);
          resultElement.innerHTML = result;
        }).catch(e => {
          console.log(`compute error ${e}`);
          console.log(`stack: ${e.stack}`);
        })
      });
    }).catch(e => {
      console.log(`compilation error ${e}`);
      console.log(`stack: ${e.stack}`);
    });
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