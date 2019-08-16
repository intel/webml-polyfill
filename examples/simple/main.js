const computeButton = document.getElementById('compute')
const inputElement1 = document.getElementById('input1');
const inputElement2 = document.getElementById('input2');
const resultElement = document.getElementById('result');

const isWebGL = document.getElementById('isWebGL');
const isWASM = document.getElementById('isWASM');
const isPREFER_SUSTAINED_SPEED = document.getElementById('isPREFER_SUSTAINED_SPEED');
const isPREFER_FAST_SINGLE_ANSWER = document.getElementById('isPREFER_FAST_SINGLE_ANSWER');
const isPREFER_LOW_POWER = document.getElementById('isPREFER_LOW_POWER');

var selectedBackend= $("input[type='radio']:checked").val();
const defaultPrefer = nn.PREFER_SUSTAINED_SPEED;
var selectedPrefer = defaultPrefer;
async function loadModelDataFile(url) {
  let response = await fetch(url);
  let bytes = await response.arrayBuffer();
  return bytes;
}

async function main() { 

    let bytes = await loadModelDataFile('model_data.bin');
    let simpleModel = new SimpleModel(bytes);
    simpleModel.createCompiledModel().then(result => {
    console.log(`compilation result: ${result}`);
    computeButton.removeAttribute('disabled');

    isWASM.addEventListener('change', e => {
        nn = navigator.ml_polyfill.getNeuralNetworkContext();
        selectedBackend= $("input[type='radio']:checked").val();
        selectedPrefer = defaultPrefer;
      })

    isWebGL.addEventListener('change', e => {
        nn = navigator.ml_polyfill.getNeuralNetworkContext();
        selectedBackend= $("input[type='radio']:checked").val();
        selectedPrefer = defaultPrefer;
      })

    isPREFER_SUSTAINED_SPEED.addEventListener('change', e => {
        nn = navigator.ml.getNeuralNetworkContext();
        selectedPrefer=nn.PREFER_SUSTAINED_SPEED;
        selectedBackend= $("input[type='radio']:checked").val();
      })

    isPREFER_FAST_SINGLE_ANSWER.addEventListener('change', e => {
        nn = navigator.ml.getNeuralNetworkContext();
        selectedPrefer=nn.PREFER_FAST_SINGLE_ANSWER;
        selectedBackend= $("input[type='radio']:checked").val();
      })

    isPREFER_LOW_POWER.addEventListener('change', e => {
        nn = navigator.ml.getNeuralNetworkContext();
        selectedPrefer=nn.PREFER_LOW_POWER;
        selectedBackend= $("input[type='radio']:checked").val();
          })

    computeButton.addEventListener('click', async function (e) {
        simpleModel = new SimpleModel(bytes);
        let result = await simpleModel.createCompiledModel();
        console.log(`compilation result: ${result}`);
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
