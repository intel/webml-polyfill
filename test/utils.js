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