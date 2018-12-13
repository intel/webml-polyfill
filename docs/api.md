# API Documentation
## Introduction
This is the API doc of WebML Polyfill. The current API is modeled from [NN API](https://developer.android.com/ndk/guides/neuralnetworks/index.html) to JavaScript. It only serves as a proof-of-concept (POC) for Web Neural Networks API proposal.
## IDL
### Navigator
```webidl
partial interface Navigator {
  readonly attribute ML ml;
};
```

### ML
```webidl
interface ML {
  NeuralNetworkContext getNeuralNetworkContext();
};
```

### NeuralNetworkContext
```webidl
interface NeuralNetworkContext {
  // Operand types.
  const long FLOAT32 = 0;
  const long INT32 = 1;
  const long UINT32 = 2;
  const long TENSOR_FLOAT32 = 3;
  const long TENSOR_INT32 = 4;
  const long TENSOR_QUANT8_ASYMM = 5;

  // Operation types.
  const long ADD = 0;
  const long AVERAGE_POOL_2D = 1;
  const long CONCATENATION = 2;
  const long CONV_2D = 3;
  const long DEPTHWISE_CONV_2D = 4;
  const long DEPTH_TO_SPACE = 5;
  const long DEQUANTIZE = 6;
  const long EMBEDDING_LOOKUP = 7;
  const long FLOOR = 8;
  const long FULLY_CONNECTED = 9;
  const long HASHTABLE_LOOKUP = 10;
  const long L2_NORMALIZATION = 11;
  const long L2_POOL_2D = 12;
  const long LOCAL_RESPONSE_NORMALIZATION = 13;
  const long LOGISTIC = 14;
  const long LSH_PROJECTION = 15;
  const long LSTM = 16;
  const long MAX_POOL_2D = 17;
  const long MUL = 18;
  const long RELU = 19;
  const long RELU1 = 20;
  const long RELU6 = 21;
  const long RESHAPE = 22;
  const long RESIZE_BILINEAR = 23;
  const long RNN = 24;
  const long SOFTMAX = 25;
  const long SPACE_TO_DEPTH = 26;
  const long SVDF = 27;
  const long TANH = 28;
  const long ATROUS_CONV_2D = 10003;
  const long ATROUS_DEPTHWISE_CONV_2D = 10004;

  // Fused activation function types.
  const long FUSED_NONE = 0;
  const long FUSED_RELU = 1;
  const long FUSED_RELU1 = 2;
  const long FUSED_RELU6 = 3;

  // Implicit padding algorithms.
  const long PADDING_SAME = 1;
  const long PADDING_VALID = 2;

  // Execution preferences.
  const long PREFER_LOW_POWER = 0;
  const long PREFER_FAST_SINGLE_ANSWER = 1;
  const long PREFER_SUSTAINED_SPEED = 2;

  Promise<Model> createModel();
};
```

### Model
```webidl
dictionary OperandOptions {
  required long type;
  sequence<unsigned long> dimensions;
  // scale: an non-negative floating point value
  float scale;
  // zeroPoint: an integer, in range [0, 255]
  long zeroPoint;
};

interface Model {
  void addOperand(OperandOptions options);
  void setOperandValue(unsigned long index, ArrayBufferView data);
  void addOperation(long type, sequence<unsigned long> inputs, sequence<unsigned long> outputs);
  void identifyInputsAndOutputs(sequence<unsigned long> inputs, sequence<unsigned long> outputs);
  Promise<long> finish();
  Promise<Compilation> createCompilation();
};
```

### Compilation
```webidl
interface Compilation {
  void setPreference(long preference);
  Promise<long> finish();
  Promise<Execution> createExecution();
};
```

### Execution
```webidl
interface Execution {
  void setInput(unsigned long index, ArrayBufferView data);
  void setOutput(unsigned long index, ArrayBufferView data);
  Promise<long> startCompute();
};
```
## Examples
Build a simple model to compute: `(tensor0 + tensor1) * (tensor2 + tensor3)`. The tensor0 and tensor2 are constants. The tensor1 and tensor3 are inputs.
```js
tensor0 ---+
           +--- ADD ---> intermediateOutput0 ---+
tensor1 ---+                                    |
                                                +--- MUL---> output
tensor2 ---+                                    |
           +--- ADD ---> intermediateOutput1 ---+
tensor3 ---+
```
### Getting NeuralNetworkContext
```js
const nn = navigator.ml.getNeuralNetworkContext();
```
### Building model
```js
const TENSOR_SIZE = 200;
let operandIndex = 0;
  
// Create a Model object.
let model = await nn.createModel();

let float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: [TENSOR_SIZE]};
let scalarInt32Type = {type: nn.INT32};

// Add the operand for the NONE activation function, and set its value to FUSED_NONE.
let fusedActivationFuncNone = operandIndex++;
model.addOperand(scalarInt32Type);
model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

// tensor0 is a constant tensor, set its value from an ArrayBuffer object.
// The ArrayBuffer object may contain the training data loaded before hand.
let tensor0 = operandIndex++;
model.addOperand(float32TensorType);
model.setOperandValue(tensor0, new Float32Array(arrayBuffer, 0, TENSOR_SIZE));

// tensor1 is one of the input tensors. Its value will be set before execution.
let tensor1 = operandIndex++;
model.addOperand(float32TensorType);

// tensor2 is another a constant tensor, set its value from same ArrayBuffer object.
let tensor2 = operandIndex++;
model.addOperand(float32TensorType);
model.setOperandValue(tensor2, new Float32Array(arrayBuffer, TENSOR_SIZE * Float32Array.BYTES_PER_ELEMENT, TENSOR_SIZE));

// tensor3 is another input tensor. Its value will be set before execution.
let tensor3 = operandIndex++;
model.addOperand(float32TensorType);

// intermediateOutput0 is the output of the first ADD operation.
let intermediateOutput0 = operandIndex++;
model.addOperand(float32TensorType);

// intermediateOutput1 is the output of the second ADD operation.
let intermediateOutput1 = operandIndex++;
model.addOperand(float32TensorType);

// multiplierOutput is the output of the MUL operation.
let multiplierOutput = operandIndex++;
model.addOperand(float32TensorType);

// Add the MUL operation. (Test operations reorder)
// Note that intermediateOutput0 and intermediateOutput1 are specified
// as inputs to the operation.
model.addOperation(nn.MUL, [intermediateOutput0, intermediateOutput1, fusedActivationFuncNone], [multiplierOutput]);

// Add the first ADD operation.
model.addOperation(nn.ADD, [tensor0, tensor1, fusedActivationFuncNone], [intermediateOutput0]);

// Add the second ADD operation.
model.addOperation(nn.ADD, [tensor2, tensor3, fusedActivationFuncNone], [intermediateOutput1]);

// Identify the input and output tensors to the model.
// Inputs: {tensor1, tensor3}
// Outputs: {multiplierOutput}
model.identifyInputsAndOutputs([tensor1, tensor3], [multiplierOutput]);

// Finish building the model.
await model.finish();
```
### Compiling model
```js
// Create a Compilation object for the constructed model.
let compilation = await model.createCompilation();

// Set the preference for the compilation as PREFER_FAST_SINGLE_ANSWER.
compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);

// Finish the compilation.
await compilation.finish();
```
### Executing model
```js
// Create an Execution object for the compiled model.
let execution = await compilation.createExecution();

// Setup the input tensors.
// They may contain data provided by user.
let inputTensor1 = new Float32Array(TENSOR_SIZE);
inputTensor1.fill(inputValue1);
let inputTensor2 = new Float32Array(TENSOR_SIZE);
inputTensor2.fill(inputValue2);

// Associate input tensors to model inputs.
execution.setInput(0, inputTensor1);
execution.setInput(1, inputTensor2);

// Associate output tensor to model output.
let outputTensor = new Float32Array(TENSOR_SIZE);
execution.setOutput(0, outputTensor);

await execution.startCompute();

// The computed result is now in outputTensor.
```
