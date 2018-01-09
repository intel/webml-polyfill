async function loadModelDataFile(url) {
  let response = await fetch(url);
  let arrayBuffer = await response.arrayBuffer();
  let bytes = new Uint8Array(arrayBuffer);
  return bytes;
}

var flatBufferModel;
function main() {
  loadModelDataFile('./model/mobilenet_v1_1.0_224.tflite').then(bytes => {
    let buf = new flatbuffers.ByteBuffer(bytes);
    flatBufferModel = tflite.Model.getRootAsModel(buf);
    printFlatBufferModel(flatBufferModel);
  }).catch(e => {
    console.log(e);
  })
}

const BuiltinOperators = [
  'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEPTH_TO_SPACE',
  'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED', 'HASHTABLE_LOOKUP',
  'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION', 'LOGISTIC',
  'LSH_PROJECTION', 'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU', 'RELU1', 'RELU6', 'RESHAPE',
  'RESIZE_BILINEAR', 'RNN', 'SOFTMAX', 'SPACE_TO_DEPTH', 'SVDF', 'TANH'];

function printFlatBufferModel(model) {
  function printOperatorCode(operatorCode) {
    console.log(`\t [builtin_code: ${BuiltinOperators[operatorCode.builtinCode()]}, custom_code: ${operatorCode.customCode()}]`);
  }
  function printSubgraph(subgraph) {
    console.log(`\t ${subgraph}`);
  }
  function printBuffer(buffer) {
    console.log(`\t [data: ${buffer.data()}, length: ${buffer.dataLength()}]`);
  }
  console.log(`version: ${model.version()}`);
  console.log(`description: ${model.description()}`);
  console.log(`operator_codes[${model.operatorCodesLength()}]:`);
  for (let i = 0; i < model.operatorCodesLength(); ++i) {
    printOperatorCode(model.operatorCodes(i));
  }
  console.log(`subgraphs[${model.subgraphsLength()}]:`);
  for (let i = 0; i < model.subgraphsLength(); ++i) {
    printSubgraph(model.subgraphs(i));
  }
  console.log(`buffers[${model.buffersLength()}]:`);
  for (let i = 0; i < model.buffersLength(); ++i) {
    printBuffer(model.buffers(i));
  }
}