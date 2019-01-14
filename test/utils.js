const nn = navigator.ml.getNeuralNetworkContext();
let options = {};
const EPISILON = 1e-5;
const EPISILON5ULP = 5.0 * 0.0009765625;
let episilonCTS = EPISILON;

function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}

function almostEqual(a, b, episilon=1e-6) {
  let delta = Math.abs(a - b);
  if (delta < episilon) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function almostEqualCTS(a, b) {
  return almostEqual(a, b, episilonCTS)
}

function setOptions() {
  // visit URL(http://domain-name/test/index.html?backend=webgl) to test Unit Tests by WebGL backend
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)backend=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  if (r != null) {
    var backend = unescape(r[2]).toLowerCase();
    if (backend === "wasm") {
      options = {
        "backend": 'WASM',
        "prefer": 'fast'
      };
    } else if (backend === "webgl") {
      options = {
        "backend": 'WebGL',
        "prefer": 'sustained'
      };
    } else if (backend === "cldnn") {
      // use PREFER_SUSTAINED_SPEED for Linux/Windows clDNN backend
      prefer = nn.PREFER_SUSTAINED_SPEED;
      options = {
        "backend": 'WebML',
        "prefer": 'sustained'
      };
    } else if (backend === "mkldnn") {
      // use PREFER_FAST_SINGLE_ANSWER for Linux/Windows mkldnn backend
      prefer = nn.PREFER_FAST_SINGLE_ANSWER;
      options = {
        "backend": 'WebML',
        "prefer": 'fast'
      };
    } else if (backend === "nnapi") {
      // use PREFER_SUSTAINED_SPEED for Android NNAPI backend
      prefer = nn.PREFER_SUSTAINED_SPEED;
      options = {
        "backend": 'WebML',
        "prefer": 'sustained'
      };
    } else if (backend === "bnns") {
      // use PREFER_FAST_SINGLE_ANSWER for MacOS BNNS backend
      options = {
        "backend": 'WebML',
        "prefer": 'fast'
      };
    } else if (backend === "mps") {
      // use PREFER_SUSTAINED_SPEED for MacOS MPS backend
      options = {
        "backend": 'WebML',
        "prefer": 'sustained'
      };
      // As MPS computes on FP16, use 5ULP of FP16 range
      episilonCTS = EPISILON5ULP;
    }
  }
}

function getPreferenceCode(preferenceStr) {
  let prefer;
  if (preferenceStr === 'sustained') {
    prefer = nn.PREFER_SUSTAINED_SPEED;
  } else if (preferenceStr === 'fast') {
    prefer = nn.PREFER_FAST_SINGLE_ANSWER;
  } else {
    console.error('Invalid preference string.');
  }
  return prefer;
}


// loadUrl and loadTensor functions are for end2end test cases
async function loadUrl(url) {
  return new Promise(resolve => {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    request.responseType = 'arraybuffer';
    request.onload = _ => {
      if (request.readyState === 4 && request.status === 200)
        resolve(new Uint8Array(request.response));
    };
    request.send();
  });
}

async function loadTensor(tensorFile) {
  let result = await loadUrl(tensorFile);
  if (onnx.TensorProto.verify(result))
    throw new Error(`Invalid tensor`);
  let tensor = onnx.TensorProto.decode(result);
  return getTensorData(tensor);
}
