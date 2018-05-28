let options = {};

function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}

function almostEqual(a, b) {
  const FLOAT_EPISILON = 1e-6;
  let delta = Math.abs(a - b);
  if (delta < FLOAT_EPISILON) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function setOptions() {
  // visit URL(http://domain-name/test/index.html?backend=webgl2) to test Unit Tests by WebGL2 backend
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)backend=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  if (r != null && unescape(r[2]) == "webgl2") {
    options = {
      "useWebGL2": true
    };
  }
}
