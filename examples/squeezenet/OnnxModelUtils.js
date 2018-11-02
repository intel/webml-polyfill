var onnx = protobuf.roots.onnx.onnx;

function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}

function getObjectByName(array, name) {
  let ret;
  array.forEach((object) => {
    if (object.name === name)
      ret = object;
  });
  return ret;
}

function getTensorData(tensor) {
  let data;
  switch (tensor.dataType) {
    case onnx.TensorProto.DataType.FLOAT: {
      if (tensor.floatData && tensor.floatData.length > 0) {
        data = new Float32Array(tensor.floatData);
      } else if (tensor.rawData && tensor.rawData.length > 0) {
        let dataView = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
        let length = tensor.dims.length ? product(tensor.dims) : 1;
        data = new Float32Array(length);
        for (let i = 0; i < length; ++i) {
          // raw data is stored in little-endian order
          data[i] = dataView.getFloat32(i*Float32Array.BYTES_PER_ELEMENT, true);
        }
      }
    } break;
    case onnx.TensorProto.DataType.INT64: {
      console.info(`Tensor ${tensor.name} has 64-bit data. Cast to a 32-bit array.`);
      if (tensor.int64Data && tensor.int64Data.length > 0) {
        data = new Int32Array(tensor.int64Data);
      } else if (tensor.rawData && tensor.rawData.length > 0) {
        let dataView = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
        let length = tensor.dims.length ? product(tensor.dims) : 1;
        data = new Int32Array(length);
        for (let i = 0; i < length; ++i)
          // raw data is stored in little-endian order
          data[i] = dataView.getInt32(i*BigInt64Array.BYTES_PER_ELEMENT, true);
      }
    } break;
    default: {
      throw new Error(`tensor type ${tensor.dataType} is not supproted.`);
    }
  }

  if (tensor.dims.length === 4) {
    // NCHW -> NHWC
    let nhwcData = new Float32Array(data.length);
    const N = tensor.dims[0];
    const C = tensor.dims[1];
    const H = tensor.dims[2];
    const W = tensor.dims[3];
    for (let n = 0; n < N; ++n) {
      for (let c = 0; c < C; ++c) {
        for (let h = 0; h < H; ++h) {
          for (let w = 0; w < W; ++w) {
            nhwcData[n*H*W*C + h*W*C + w*C + c] = data[n*C*H*W + c*H*W + h*W + w];
          }
        }
      }
    }
    data = nhwcData;
  }
  return data;
}

async function loadOnnxModel(modelName) {
  let response = await fetch(modelName);
  let bytes = await response.arrayBuffer();
  let buffer = new Uint8Array(bytes);
  let err = onnx.ModelProto.verify(buffer);
  if (err) {
    throw new Error(`Invalid model ${err}`);
  }
  let modelProto = onnx.ModelProto.decode(buffer);
  return modelProto;
}

function printOnnxModel(model) {
  console.log(`Print ONNX model:`);
  console.log(model);

  function printNode(node) {
    console.log(`    {opType: ${node.opType}, input: [${node.input}], output: [${node.output}]}`);
    switch(node.opType) {
      case 'Conv': {
        let attributes = node.attribute;
        let attr = null;
        if (attr = getObjectByName(attributes, 'kernel_shape')) {
          console.log(`    kernel_shape: [${attr.ints}]`);
        }
        if (attr = getObjectByName(attributes, 'pads')) {
          console.log(`    pads: [${attr.ints}]`);
        }
        if (attr = getObjectByName(attributes, 'strides')) {
          console.log(`    strides: [${attr.ints}]`);
        }
      } break;
      case 'Relu': {} break;
      case 'AveragePool':
      case 'MaxPool': {
        let attributes = node.attribute;
        let attr = null;
        if (attr = getObjectByName(attributes, 'kernel_shape')) {
          console.log(`    kernel_shape: [${attr.ints}]`);
        }
        if (attr = getObjectByName(attributes, 'pads')) {
          console.log(`    pads: [${attr.ints}]`);
        }
        if (attr = getObjectByName(attributes, 'strides')) {
          console.log(`    strides: [${attr.ints}]`);
        }
      } break;
      case 'Concat': {} break;
      case 'Dropout': {} break;
      case 'GlobalAveragePool': {} break;
      case 'Softmax': {} break;
      case 'BatchNormalization': {} break;
      case 'Add': {} break;
      case 'Mul': {} break;
      case 'Constant': {} break;
      case 'Reshape': {} break;
      case 'Gemm': {} break;
      case 'Unsqueeze': {} break;
      default: {
        console.warn(`    ${node.opType} is not supported.}`);
      }
    }
  }
  function printValueInfo(valueInfo) {
    console.log(`    name: ${valueInfo.name}`);
    let tensorType = valueInfo.type.tensorType;
    console.log(`    type: {elemType: ${tensorType.elemType}, shape: [${tensorType.shape.dim.map(dim => {return dim.dimValue;})}]}`);
  }
  function printTensor(tensor) {
    console.log(`    {name: ${tensor.name}, dataType: ${tensor.dataType}, dims: [${tensor.dims}]}`);
    let data;
    if (tensor.dataType == onnx.TensorProto.DataType.FLOAT) {
      if (tensor.floatData && tensor.floatData.length > 0) {
        data = tensor.floatData;
      } else if (tensor.rawData && tensor.rawData.length > 0) {
        let dataView = new DataView(tensor.rawData.buffer, tensor.rawData.byteOffset, tensor.rawData.byteLength);
        let length = product(tensor.dims);
        data = new Float32Array(length);
        for (let i = 0; i < length; ++i) {
          // raw data is stored in little-endian order
          data[i] = dataView.getFloat32(i*Float32Array.BYTES_PER_ELEMENT, true);
        }
      }
    }
    //console.log(`    data(${data.length}): [${data}]`);
  }
  function printGraph(graph) {
    console.log(`name: ${graph.name}`);
    console.log(`node(${graph.node.length}):`);
    for (let i = 0; i < graph.node.length; ++i) {
      console.log(`  node[${i}]:`);
      printNode(graph.node[i]);
    }
    console.log(`input(${graph.input.length}):`);
    for (let i = 0; i < graph.input.length; ++i) {
      console.log(`  input[${i}]:`);
      printValueInfo(graph.input[i]);
    }
    console.log(`output(${graph.output.length}):`);
    for (let i = 0; i < graph.output.length; ++i) {
      console.log(`  output[${i}]:`);
      printValueInfo(graph.output[i]);
    }
    console.log(`initializer(${graph.initializer.length}):`);
    for (let i = 0; i < graph.initializer.length; ++i) {
      console.log(`  initializer[${i}]:`);
      printTensor(graph.initializer[i]);
    }
  }

  console.log(`irVersion: ${model.irVersion}`);
  console.log(`producerName: ${model.producerName}`);
  printGraph(model.graph);
};