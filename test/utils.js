let options = {};

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
  return almostEqual(a, b, episilon=1e-5)
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
