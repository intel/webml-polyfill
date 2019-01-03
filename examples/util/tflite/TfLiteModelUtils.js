const BuiltinOperators = [
  'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D', 'DEPTH_TO_SPACE',
  'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED', 'HASHTABLE_LOOKUP',
  'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION', 'LOGISTIC',
  'LSH_PROJECTION', 'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU', 'RELU1', 'RELU6', 'RESHAPE',
  'RESIZE_BILINEAR', 'RNN', 'SOFTMAX', 'SPACE_TO_DEPTH', 'SVDF', 'TANH', "CONCAT_EMBEDDINGS",
  "SKIP_GRAM", "CALL", "CUSTOM", "EMBEDDING_LOOKUP_SPARSE", "PAD", "UNIDIRECTIONAL_SEQUENCE_RNN",
  "GATHER", "BATCH_TO_SPACE_ND", "SPACE_TO_BATCH_ND", "TRANSPOSE", "MEAN", "SUB", "DIV", 'SQUEEZE'];

const TensorTypes = ['FLOAT32', 'FLOAT16', 'INT32', 'UINT8', 'INT64', 'STRING'];

const ActivationFunctionTypes = ['NONE', 'RELU', 'RELU1', 'RELU6', 'TANH', 'SIGN_BIT'];

const PaddingTypes = ['SAME', 'VALID'];

function printTfLiteModel(model) {
  function printOperatorCode(operatorCode, i) {
    console.log(`\t operator_codes[${i}]: {builtin_code: ${BuiltinOperators[operatorCode.builtinCode()]}, custom_code: ${operatorCode.customCode()}}`);
  }
  function printTensor(tensor, i) {
    console.log(`\t\t tensors[${i}]: `+
      `{name: ${tensor.name()}, type: ${TensorTypes[tensor.type()]}, shape: [${tensor.shapeArray()}], buffer: ${tensor.buffer()}}`)
  }
  function printOperator(operator, i) {
    let op = BuiltinOperators[model.operatorCodes(operator.opcodeIndex()).builtinCode()];
    console.log(`\t\t operators[${i}]: `);
    console.log(`\t\t\t {opcode: ${op}, inputs: [${operator.inputsArray()}], outputs: [${operator.outputsArray()}], `)
    switch(op) {
      case 'ADD': {
        let options = operator.builtinOptions(new tflite.AddOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'CONV_2D': {
        let options = operator.builtinOptions(new tflite.Conv2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${PaddingTypes[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `dilation_w: ${options.dilationWFactor()}, `+
          `dilation_h: ${options.dilationHFactor()}, ` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'DEPTHWISE_CONV_2D': {
        let options = operator.builtinOptions(new tflite.DepthwiseConv2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${PaddingTypes[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, `+
          `stride_h: ${options.strideH()}, ` +
          `dilation_w: ${options.dilationWFactor()}, `+
          `dilation_h: ${options.dilationHFactor()}, ` +
          `depth_multiplier: ${options.depthMultiplier()}, ` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'AVERAGE_POOL_2D': {
        let options = operator.builtinOptions(new tflite.Pool2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${PaddingTypes[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `filter_width: ${options.filterWidth()}, ` +
          `filter_height: ${options.filterHeight()}, ` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'SOFTMAX': {
        let options = operator.builtinOptions(new tflite.SoftmaxOptions());
        console.log(`\t\t\t  builtin_options: {beta: ${options.beta()}}}`);
      } break;
      case 'RESHAPE': {
        let options = operator.builtinOptions(new tflite.ReshapeOptions());
        console.log(`\t\t\t  builtin_options: {new_shape: [${options.newShapeArray()}]}}`);
      } break;
      case 'MAX_POOL_2D': {
        let options = operator.builtinOptions(new tflite.Pool2DOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `padding: ${PaddingTypes[options.padding()]}, ` +
          `stride_w: ${options.strideW()}, ` +
          `stride_h: ${options.strideH()}, ` +
          `filter_width: ${options.filterWidth()}, ` +
          `filter_height: ${options.filterHeight()}, ` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'CONCATENATION': {
        let options = operator.builtinOptions(new tflite.ConcatenationOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `axis: ${options.axis()}, ` +
          `fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'SQUEEZE': {
        let options = operator.builtinOptions(new tflite.SqueezeOptions());
        console.log(`\t\t\t  builtin_options: {` +
          `squeezeDims: ${options.squeezeDims()}, ` +
          `squeezeDimsLength: ${options.squeezeDimsLength()}, ` +
          `squeezeDimsArray: ${options.squeezeDimsArray()}}}`);
      } break;
      case 'FULLY_CONNECTED': {
        let options = operator.builtinOptions(new tflite.FullyConnectedOptions());
        console.log(`\t\t\t  builtin_options: {fused_activation_function: ${ActivationFunctionTypes[options.fusedActivationFunction()]}}}`);
      } break;
      case 'RESIZE_BILINEAR': {
      } break;
      default: {
        console.warn(`\t\t\t  builtin_options: ${op} is not supported.}`);
      }
    }
  }
  function printSubgraph(subgraph, i) {
    console.log(`  subgraphs[${i}]`);
    console.log(`\t name: ${subgraph.name()}`);
    console.log(`\t inputs: [${subgraph.inputsArray()}]`);
    console.log(`\t outputs: [${subgraph.outputsArray()}]`);
    console.log(`\t tensors(${subgraph.tensorsLength()}):`)
    for (let i = 0; i < subgraph.tensorsLength(); ++i) {
      printTensor(subgraph.tensors(i), i);
    }
    console.log(`\t operators(${subgraph.operatorsLength()}):`)
    for (let i = 0; i < subgraph.operatorsLength(); ++i) {
      printOperator(subgraph.operators(i), i);
    }
  }
  function printBuffer(buffer, i) {
    console.log(`\t buffer[${i}]: {data: ${buffer.data()}, length: ${buffer.dataLength()}}`);
  }
  console.log(`version: ${model.version()}`);
  console.log(`description: ${model.description()}`);
  console.log(`operator_codes(${model.operatorCodesLength()}):`);
  for (let i = 0; i < model.operatorCodesLength(); ++i) {
    printOperatorCode(model.operatorCodes(i), i);
  }
  console.log(`subgraphs(${model.subgraphsLength()}):`);
  for (let i = 0; i < model.subgraphsLength(); ++i) {
    printSubgraph(model.subgraphs(i), i);
  }
  console.log(`buffers[${model.buffersLength()}]:`);
  for (let i = 0; i < model.buffersLength(); ++i) {
    printBuffer(model.buffers(i), i);
  }
}