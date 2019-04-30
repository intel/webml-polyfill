const nn = navigator.ml.getNeuralNetworkContext();
const assert = chai.assert;
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

function almostEqualRM(a, b) {
  let delta = Math.abs(a - b);
  // refer to https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/model/__init__.py#L49
  if (delta <= 1e-7 + 1e-3 * Math.abs(b)) {
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
  // visit URL(http://domain-name/test/index.html?prefer=fast/sustained/low)
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)prefer=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  var macosPlatforms = ['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K'];
  if (r != null) {
    var prefer = unescape(r[2]).toLowerCase();
    if(navigator.ml.isPolyfill) {
      if (prefer === "fast") {
        options = {
          "backend": 'WASM',
          "prefer": 'fast'
        };
      } else if (prefer === "sustained") {
        options = {
          "backend": 'WebGL',
          "prefer": 'sustained'
        };
      }
    } else {
      if (prefer === "sustained") {
        // use PREFER_SUSTAINED_SPEED for Linux/Windows clDNN backend
        prefer = nn.PREFER_SUSTAINED_SPEED;
        options = {
          "backend": 'WebML',
          "prefer": 'sustained'
        };
        // As MPS computes on FP16, use 5ULP of FP16 range
        if (macosPlatforms.indexOf(navigator.platform) !== -1) {
          episilonCTS = EPISILON5ULP;
        }
      } else if (prefer === "fast") {
        // use PREFER_FAST_SINGLE_ANSWER for Linux/Windows mkldnn backend
        prefer = nn.PREFER_FAST_SINGLE_ANSWER;
        options = {
          "backend": 'WebML',
          "prefer": 'fast'
        };
      } else if (prefer === "low") {
        prefer = nn.PREFER_LOW_POWER;
        options = {
          "backend": 'WebML',
          "prefer": 'low'
        };
      }
    }
  }
}

function getPreferenceCode(preferenceStr) {
  let prefer;
  if (preferenceStr === 'sustained') {
    prefer = nn.PREFER_SUSTAINED_SPEED;
  } else if (preferenceStr === 'fast') {
    prefer = nn.PREFER_FAST_SINGLE_ANSWER;
  } else if (preferenceStr === 'low') {
    prefer = nn.PREFER_LOW_POWER;
  }else {
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

async function assertThrowsAsync(fn, regExp) {
  let f = () => {};
  try {
    await fn();
  } catch(e) {
    f = () => {throw e};
  } finally {
    assert.throws(f, regExp);
  }
}

async function assertDoesNotThrowAsync(fn, regExp) {
  let f = () => {};
  try {
    await fn();
  } catch(e) {
    f = () => {throw e};
  } finally {
    assert.doesNotThrow(f, regExp);
  }
}
