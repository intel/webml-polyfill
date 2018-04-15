import ndarray from 'ndarray'
import ops from 'ndarray-ops'

/**
 * Create indicesForReshaped for 2D reshaped tensor
 *
 * @param {number[]} shape
 * @param {boolean} square
 * @param {number} axis
 */
export function createIndicesFor2DReshaped(shape, square = false, axis = -1) {
  const size = shape.reduce((a, b) => a * b, 1);
  const indicesArr = ndarray(new Int32Array(size), shape);

  if (square) {
    // called by Tensor.reshapeTo2DSquare
    const squareDim = Math.ceil(Math.sqrt(size));
    const indicesRowArrReshaped = ndarray(new Int32Array(squareDim ** 2), [squareDim, squareDim]);
    const indicesColArrReshaped = ndarray(new Int32Array(squareDim ** 2), [squareDim, squareDim]);
    const indicesArrReshaped = ndarray(new Int32Array(squareDim ** 2), [squareDim, squareDim]);
    for (let i = 0; i < squareDim; i++) {
      ops.assigns(indicesRowArrReshaped.pick(i, null), i);
    }
    for (let j = 0; j < squareDim; j++) {
      ops.assigns(indicesColArrReshaped.pick(null, j), j);
    }
    // i * cols + j
    ops.muls(indicesArrReshaped, indicesRowArrReshaped, squareDim);
    ops.addeq(indicesArrReshaped, indicesColArrReshaped);
    indicesArr.data.set(indicesArrReshaped.data.subarray(0, indicesArr.size));
  } else {
    // called by Tensor.reshapeTo2D
    if (axis < 0) {
      axis = shape.length + axis;
    }
    const axisSize = shape[axis];
    const indicesRowArr = ndarray(new Int32Array(size), shape);
    const indicesColArr = ndarray(new Int32Array(size), shape);
    const otherAxes = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    const otherAxesSize = otherAxes.reduce((a, b) => a * b, 1);
    let tmp = [];
    for (let i = 0; i < otherAxesSize; ++i) {
      tmp.push(i);
    }
    const indicesRowArrSlice = ndarray(new Int32Array(tmp), otherAxes);
    const axisSlices = Array(shape.length).fill(null);
    for (let n = 0; n < axisSize; n++) {
      axisSlices[axis] = n;
      ops.assign(indicesRowArr.pick(...axisSlices), indicesRowArrSlice);
      ops.assigns(indicesColArr.pick(...axisSlices), n);
    }
    // i * cols + j
    ops.muls(indicesArr, indicesRowArr, axisSize);
    ops.addeq(indicesArr, indicesColArr);
  }

  return indicesArr;
}
