const nn = navigator.ml.getNeuralNetworkContext();
let options = {};
let prefer = nn.PREFER_FAST_SINGLE_ANSWER;
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
        "backend": 'WASM'
      };
    } else if (backend === "webgl") {
      options = {
        "backend": 'WebGL'
      };
    } else if (backend === "cldnn") {
      // use PREFER_SUSTAINED_SPEED for Linux/Windows clDNN backend
      prefer = nn.PREFER_SUSTAINED_SPEED;
    } else if (backend === "bnns") {
      // use PREFER_FAST_SINGLE_ANSWER for MacOS BNNS backend
      prefer = nn.PREFER_FAST_SINGLE_ANSWER;
    } else if (backend === "mps") {
      // use PREFER_SUSTAINED_SPEED for MacOS MPS backend
      prefer = nn.PREFER_SUSTAINED_SPEED;
      // As MPS computes on FP16, use 5ULP of FP16 range
      episilonCTS = EPISILON5ULP;
    }
  }
}
