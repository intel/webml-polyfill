const nn = navigator.ml.getNeuralNetworkContext();
let options = {};
let prefer = nn.PREFER_FAST_SINGLE_ANSWER;
let episilonCTS = 1e-5;

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
  // visit URL(http://domain-name/test/index.html?backend=webgl2) to test Unit Tests by WebGL2 backend
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)backend=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  if (r != null) {
    var backend = unescape(r[2]);
    if (backend == "webgl2") {
      options = {
        "useWebGL2": true
      };
    } else if (backend == "bnns") {
      // use PREFER_FAST_SINGLE_ANSWER for MacOS BNNS backend
      prefer = nn.PREFER_FAST_SINGLE_ANSWER;
    } else if (backend == "mps") {
      // use PREFER_SUSTAINED_SPEED for MacOS MPS backend
      prefer = nn.PREFER_SUSTAINED_SPEED;
      episilonCTS = 5.0 * 0.0009765625;
    }
  }
}
