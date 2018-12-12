describe('Operations Test', function() {
  const assert = chai.assert;

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

  it('addFloat32', function() {
    const value1 = 0.5;
    const value2 = 0.5;
    const dims = [1, 227, 227, 3];
  
    let shape1 = new Module.Shape;
    shape1.type = Module.TENSOR_FLOAT32;
    shape1.dimensions = dims;
    let tensor1Data = new Float32Array(product(shape1.dimensions));
    tensor1Data.fill(value1);
    let tensor1ByteLength = tensor1Data.length * tensor1Data.BYTES_PER_ELEMENT;
    let tensor1 = Module._malloc(tensor1ByteLength);
    Module.HEAPF32.set(tensor1Data, tensor1 >> 2);
    
    let shape2 = new Module.Shape;
    shape2.type = Module.TENSOR_FLOAT32;
    shape2.dimensions = dims;
    let tensor2Data = new Float32Array(product(shape2.dimensions));
    tensor2Data.fill(value2);
    let tensor2ByteLength = tensor2Data.length * tensor2Data.BYTES_PER_ELEMENT;
    let tensor2 = Module._malloc(tensor2ByteLength);
    Module.HEAPF32.set(tensor2Data, tensor2 >> 2);

    let shape3 = new Module.Shape;
    shape3.type = Module.TENSOR_FLOAT32;
    shape3.dimensions = [0, 0, 0, 0];

    assert.isTrue(Module.addMulPrepare(shape1, shape2, shape3));
    assert.deepEqual(shape3.dimensions, dims);

    let tensor3Length = product(shape3.dimensions);
    let tensor3ByteLength = tensor3Length * Float32Array.BYTES_PER_ELEMENT;
    let tensor3 = Module._malloc(tensor3ByteLength);

    let start = performance.now();
    assert.isTrue(Module.addFloat32(tensor1, shape1, tensor2, shape2, Module.NONE, tensor3, shape3));
    let end = performance.now();
    console.log(`addFloat32 elapsed time: ${end - start} ms`);

    let tensor3Data = new Float32Array(Module.HEAP8.buffer, tensor3, tensor3Length);

    for (let i = 0; i < 10; ++i) {
      assert.isTrue(almostEqual(tensor3Data[i], tensor1Data[i] + tensor2Data[i]));
    }

    for (let i = tensor3Data.length - 10; i < tensor3Data.length; ++i) {
      assert.isTrue(almostEqual(tensor3Data[i], tensor1Data[i] + tensor2Data[i]));
    }

    Module._free(tensor1);
    Module._free(tensor2);
    Module._free(tensor3);
    shape1.delete();
    shape2.delete();
    shape3.delete();
  });

  it('mulFloat32', function() {
    const value1 = 0.5;
    const value2 = 0.5;
    const dims = [1, 227, 227, 3];
  
    let shape1 = new Module.Shape;
    shape1.type = Module.TENSOR_FLOAT32;
    shape1.dimensions = dims;
    let tensor1Data = new Float32Array(product(shape1.dimensions));
    tensor1Data.fill(value1);
    let tensor1ByteLength = tensor1Data.length * tensor1Data.BYTES_PER_ELEMENT;
    let tensor1 = Module._malloc(tensor1ByteLength);
    Module.HEAPF32.set(tensor1Data, tensor1 >> 2);
    
    let shape2 = new Module.Shape;
    shape2.type = Module.TENSOR_FLOAT32;
    shape2.dimensions = dims;
    let tensor2Data = new Float32Array(product(shape2.dimensions));
    tensor2Data.fill(value2);
    let tensor2ByteLength = tensor2Data.length * tensor2Data.BYTES_PER_ELEMENT;
    let tensor2 = Module._malloc(tensor2ByteLength);
    Module.HEAPF32.set(tensor2Data, tensor2 >> 2);

    let shape3 = new Module.Shape;
    shape3.type = Module.TENSOR_FLOAT32;
    shape3.dimensions = [0, 0, 0, 0];

    assert.isTrue(Module.addMulPrepare(shape1, shape2, shape3));
    assert.deepEqual(shape3.dimensions, dims);

    let tensor3Length = product(shape3.dimensions);
    let tensor3ByteLength = tensor3Length * Float32Array.BYTES_PER_ELEMENT;
    let tensor3 = Module._malloc(tensor3ByteLength);

    let start = performance.now();
    assert.isTrue(Module.mulFloat32(tensor1, shape1, tensor2, shape2, Module.NONE, tensor3, shape3));
    let end = performance.now();
    console.log(`mulFloat32 elapsed time: ${end - start} ms`);

    let tensor3Data = new Float32Array(Module.HEAP8.buffer, tensor3, tensor3Length);

    for (let i = 0; i < 10; ++i) {
      assert.isTrue(almostEqual(tensor3Data[i], tensor1Data[i] * tensor2Data[i]));
    }

    for (let i = tensor3Data.length - 10; i < tensor3Data.length; ++i) {
      assert.isTrue(almostEqual(tensor3Data[i], tensor1Data[i] * tensor2Data[i]));
    }

    Module._free(tensor1);
    Module._free(tensor2);
    Module._free(tensor3);
    shape1.delete();
    shape2.delete();
    shape3.delete();
  });

  function randomFill(data) {
    for (let i = 0; i < data.length; ++i) {
      data[i] = Math.random() * 2 - 1
    }
  }

  function product(array) {
    return array.reduce((accumulator, currentValue) => accumulator * currentValue);
  }

  it('convFloat32', function() {
    const padding = 0;
    const stride = 4;
    
    let inputShape = new Module.Shape;
    inputShape.type = Module.TENSOR_FLOAT32;
    inputShape.dimensions = [1, 227, 227, 3];

    let filterShape = new Module.Shape;
    filterShape.type = Module.TENSOR_FLOAT32;
    filterShape.dimensions = [96, 11, 11, 3];

    let biasShape = new Module.Shape;
    biasShape.type = Module.TENSOR_FLOAT32;
    biasShape.dimensions = [96];

    let outputShape = new Module.Shape;

    assert.isTrue(Module.convPrepare(inputShape, filterShape, biasShape, padding, padding, padding, padding, stride, stride, outputShape));

    assert.equal(outputShape.type, Module.TENSOR_FLOAT32);
    assert.deepEqual(outputShape.dimensions, [1, 55, 55, 96]);

    let inputData = new Float32Array(product(inputShape.dimensions));
    randomFill(inputData);
    let inputDataByteLength = inputData.length * inputData.BYTES_PER_ELEMENT;
    let input = Module._malloc(inputDataByteLength);
    Module.HEAPF32.set(inputData, input >> 2);

    let filterData = new Float32Array(product(filterShape.dimensions));
    randomFill(filterData);
    let filterDataByteLength = filterData.length * filterData.BYTES_PER_ELEMENT;
    let filter = Module._malloc(filterDataByteLength);
    Module.HEAPF32.set(filterData, filter >> 2);

    let biasData = new Float32Array(product(biasShape.dimensions));
    randomFill(biasData);
    let biasDataByteLength = biasData.length * biasData.BYTES_PER_ELEMENT;
    let bias = Module._malloc(biasDataByteLength);
    Module.HEAPF32.set(biasData, bias >> 2);

    let output = Module._malloc(product(outputShape.dimensions)*Float32Array.BYTES_PER_ELEMENT);

    let start = performance.now();
    assert.isTrue(Module.convFloat32(input, inputShape, filter, filterShape, bias, biasShape, padding, padding, padding, padding, stride, stride, Module.NONE, output, outputShape));
    let end = performance.now();
    console.log(`convFloat32 elapsed time: ${end - start} ms`);

    Module._free(input);
    Module._free(filter);
    Module._free(bias);
    Module._free(output);
    inputShape.delete();
    filterShape.delete();
    biasShape.delete();
    outputShape.delete();
  });
});