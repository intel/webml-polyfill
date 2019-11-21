import getNNOpsInstance from './NNOps'
import { OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode, OperandLifetime } from '../Enums'
import * as utils from '../utils'
import { product, findKey } from '../utils';
import Graph from '../GraphUtils';
import CyclicProfiler from '../instrument';

var warmUpRuns = 1;

export default class PreparedModel {
  constructor() {
    this._nnNative = navigator.ml.getNeuralNetworkContext();
    this._supportedOps = new Set([]);
    this._operations = [];
    this._operands = [];
    this._prepared = false;
    this._nn_ops = null;
    this._model;
    this._subgraphs = [];
    this._preference = PreferenceCode.FAST_SINGLE_ANSWER;
    this._toDelete = {
      tensorValue: [],
      tensorShape: []
    };
    this._profiler = null;
  }

  /**
   * Prepare for model execution.
   *
   * @param {Object} model - A model object built by user.
   */
  async prepare(model) {
    this._model = model;
    const modelInputs = model._inputs;
    const operations = model._operations;
    this._nn_ops = await getNNOpsInstance();

    this._preference = model._preference;
    this._supportedOps = model._supportedOps;
    this._eager = model._eager;

    if (model._operands[modelInputs[0]].type === OperandCode.TENSOR_QUANT8_ASYMM) {
        this._nn_ops.set_gemm_context_threads_num(1);
    }

    this._nn_ops.set_cpu_context_threads_num(1);

    // allocate runtime operands
    for (let i = 0; i < model._operands.length; ++i) {
      const operand = model._operands[i];
      const runtimeOperand = {};
      runtimeOperand.type = operand.type;
      runtimeOperand.dimensions = operand.dimensions;
      if (utils.isTensor(operand.type)) {
        runtimeOperand.value = this._allocateTensor(operand);
        runtimeOperand.runtimeshape = this._allocateRuntimeShape(operand);
        runtimeOperand.scale = operand.scale;
        runtimeOperand.zeroPoint = operand.zeroPoint;
        this._toDelete.tensorValue.push(runtimeOperand.value);
        this._toDelete.tensorShape.push(runtimeOperand.runtimeshape);
      } else {
        runtimeOperand.value = operand.value;
      }
      this._operands.push(runtimeOperand);
    }

    const graph = new Graph(operations.length);
    operations.forEach((op, i) => {
      graph.addNode(i, op.inputs, op.outputs);
      if (!this._supportedOps.has(op.type)) {
        // mark unsupported ops black
        graph.setBlack(i);
      }
    });
    graph.identifyInputOutputTensors(model._inputs, model._outputs);
    const partitions = graph.partition(this._eager);

    for (const {nodes, inTensors, outTensors} of partitions) {

      // Test if the first op in the partition (nodes[0]) is supported natively
      const isSupportedByNN = this._supportedOps.has(operations[nodes[0]].type);

      // summary of the partiton. e.g. "CONV x 5, ADD x 2, MUL x 2"
      const summary = utils.stringifySubgraphCompact(model, nodes);
      const backendName = isSupportedByNN ? 'WebNN' : 'WASM';
      this._subgraphs.push({
        backend: backendName,
        summary: summary,
      });

      if (!isSupportedByNN) {

        // break up a group of WASM ops to be eagerly execueted
        for (const operationId of nodes) {
          const operation = model._operations[operationId];
          this._operations.push(operation);
        }

      } else {

        // create WebNN model
        const {model, compilation, execution} =
            await this._createSubModel(nodes, inTensors, outTensors);

        // add the WEBNN_SUBGRAPH pseudo op
        this._operations.push({
          type: OperationCode.WEBNN_SUBGRAPH,
          summary: summary,
          inputs: inTensors,
          outputs: outTensors,
          model: model,             // avoid GC   intel/webml-polyfill#669
          compilation: compilation, // avoid GC   intel/webml-polyfill#669
          execution: execution,
        });
      }
    }

    this._profiler = new CyclicProfiler(this._operations.length, warmUpRuns);
    this._prepared = true;
  }

  /**
   * Launches an asynchronous execution on a prepared model.
   *
   * @param {Array} inputs - Inputs provided by user.
   * @param {Array} outputs - Outputs will receive results.
   */
  async execute(inputs, outputs) {
    if (!this._prepared) {
      throw new Error('Model is not prepared');
    }

    inputs.forEach(input => {
      const operand = this._operands[input.index];
      this._setTensorData(operand.type, operand.value, input.buffer);
    });

    for (const operation of this._operations) {
      this._profiler.startEvent();
      await this._executeOperation(operation);
      this._profiler.endEvent();
    }

    outputs.forEach((output) => {
      const operand = this._operands[output.index];
      this._getTensorData(operand.type, operand.value, output.buffer);
    });
  }

  async _createSubModel(nodes, inTensors, outTensors) {

    // create a WebNN model
    const submodel = await this._nnNative.createModel();

    // since tensorId of a subgraph should start from 0, we use it to
    // maintain a mapping from global tensor Id to local tensor Id
    const globalIdToLocalId = {};

    // counter for local tensor Id
    let operandIndex = 0;

    for (const operationId of nodes) {
      const operation = this._model._operations[operationId];

      // allocate input and output tensors for each operation
      for (const tensorId of [...operation.inputs, ...operation.outputs]) {
        const globalTensorId = parseInt(tensorId);

        // E.g., tensor A -> Node 1 -> tensor B -> Node 2 -> tensor C
        // At the time of Node 2, its input tensor B may have already been
        // allocated by the time Node 1 was processed. So we check if the
        // `globalTensorId` is already in the map.
        if (!globalIdToLocalId.hasOwnProperty(globalTensorId)) {
          const localTensorId = operandIndex++;
          globalIdToLocalId[globalTensorId] = localTensorId;
          const operand = this._model._operands[globalTensorId];
          const operandType = {
            type: operand.type,
            dimensions: operand.dimensions,
            scale: operand.scale,
            zeroPoint: operand.zeroPoint,
          };
          submodel.addOperand(operandType);
          if (operand.value) {
            submodel.setOperandValue(localTensorId, operand.value);
          }
        }
      }

      // add the operation to the submodel
      const operationInputs = operation.inputs.map(i => globalIdToLocalId[i]);
      const operationOutputs = operation.outputs.map(i => globalIdToLocalId[i]);
      submodel.addOperation(operation.type, operationInputs, operationOutputs);
    }

    // indentify the input and output tensors of the submodel
    const submodelInputs = inTensors.map(i => globalIdToLocalId[i]);
    const submodelOutputs = outTensors.map(i => globalIdToLocalId[i]);
    submodel.identifyInputsAndOutputs(submodelInputs, submodelOutputs);
    await submodel.finish();

    const compilation = await submodel.createCompilation();
    compilation.setPreference(this._preference);
    await compilation.finish();

    const execution = await compilation.createExecution();

    // bind input and output tensor buffers at compile time
    inTensors.forEach((tensorId, i) => {
      const operand = this._operands[tensorId];
      const length = product(operand.dimensions);
      const view = this._getTensorDataView(operand.type, operand.value, length);
      execution.setInput(i, view);
    });
    outTensors.forEach((tensorId, i) => {
      const operand = this._operands[tensorId];
      const length = product(operand.dimensions);
      const view = this._getTensorDataView(operand.type, operand.value, length);
      execution.setOutput(i, view);
    });

    return {model: submodel, compilation: compilation, execution: execution};
  }

  async _executeOperation(operation) {
    const nn_ops = this._nn_ops;
    let op = operation.type;
    let inputs = operation.inputs;
    let outputs = operation.outputs;
    let operands = this._operands;
    let modelOperands = this._model._operands;

    function allParametersPresent(requiredIns, requiredOuts) {
      function verify(requiredCount, indexes, type) {
        let actualCount = indexes.length;
        if (requiredCount !== actualCount) {
          throw new Error(`Operation ${op} requires ${requiredCount} ${type} operands, but got ${actualCount}.`);
        }
        indexes.forEach(index => {
          if (operands[index].value === null || operands[index].lifetime === OperandLifetime.NO_VALUE) {
            throw new Error(`Operation ${op} ${type} operand ${index} is required but missing.`);
          }
        })
      }
      verify(requiredIns, inputs, 'in');
      verify(requiredOuts, outputs, 'out');
    }

    function calculateExplicitPadding(inSize, stride, filterSize, dilationFactor, paddingCode) {
      let paddingHead = 0;
      let paddingTail = 0;

      let dilatedFilterSize = dilationFactor * (filterSize - 1) + 1;

      if (paddingCode === PaddingCode.SAME) {
        let outSize = Math.floor((inSize + stride - 1) / stride);
        let tmp = Math.floor((outSize - 1) * stride + dilatedFilterSize);
        if (tmp > inSize) {
          paddingHead = Math.floor((tmp - inSize) / 2);
          paddingTail = Math.floor((tmp - inSize) - paddingHead);
        }
      }

      return [paddingHead, paddingTail];
    }

    function calculateActivationRange(activation, output) {
      if (output.type === OperandCode.TENSOR_FLOAT32) {
        // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#261
        let float_activation_min, float_activation_max;
        if (activation === FuseCode.RELU) {
          float_activation_min = 0.0;
          float_activation_max = nn_ops.FLOAT_MAX;
        } else if (activation === FuseCode.RELU6) {
          float_activation_min = 0.0;
          float_activation_max = 6.0;
        } else if (activation === FuseCode.RELU1) {
          float_activation_min = -1.0;
          float_activation_max = 1.0;
        } else if (activation === FuseCode.NONE) {
          float_activation_min = nn_ops.FLOAT_LOWEST;
          float_activation_max = nn_ops.FLOAT_MAX;
        } else {
          throw new Error("Unsupported fused activation function.");
        }
        return [float_activation_min, float_activation_max, 0, 0];
      } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
        // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#230
        let quantized_activation_min, quantized_activation_max;
        let scale = output.scale;
        let zero_point = output.zeroPoint;

        let quantize = function(f) {
            return zero_point + Math.round(f / scale);
        };

        if (activation == FuseCode.RELU) {
          quantized_activation_min = Math.max(nn_ops.UINT8_MIN, quantize(0.0));
          quantized_activation_max = nn_ops.UINT8_MAX;
        } else if (activation == FuseCode.RELU6) {
          quantized_activation_min = Math.max(nn_ops.UINT8_MIN, quantize(0.0));
          quantized_activation_max = Math.min(nn_ops.UINT8_MAX, quantize(6.0));
        } else if (activation == FuseCode.RELU1) {
          quantized_activation_min = Math.max(nn_ops.UINT8_MIN, quantize(-1.0));
          quantized_activation_max = Math.min(nn_ops.UINT8_MAX, quantize(1.0));
        } else if (activation == FuseCode.NONE){
          quantized_activation_min = nn_ops.UINT8_MIN;
          quantized_activation_max = nn_ops.UINT8_MAX;
        } else {
          throw new Error("Unsupported fused activation function.");
        }
        return [0.0, 0.0, quantized_activation_min, quantized_activation_max];
      } else {
        throw new Error("Unsupported type of tensor for fused activation function.");
      }
    }

    // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#153
    function QuantizeMultiplier(double_multiplier) {
      let quantized_multiplier, shift;
      if (double_multiplier == 0.) {
          quantized_multiplier = 0;
          shift = 0;
          return [quantized_multiplier, shift];
      }
      let q;
      [q, shift] = frexp(double_multiplier);
      quantized_multiplier = Math.round(q * -(1 << 31));
      OPS_CHECK(quantized_multiplier <= -(1 << 31));
      if (quantized_multiplier == -(1 << 31)) {
        quantized_multiplier /= 2;
        ++shift;
      }
      OPS_CHECK(quantized_multiplier <= nn_ops.INT32_MAX);
      return [quantized_multiplier, shift];
    }

    // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#213
    function GetQuantizedConvolutionMultipler(input_scale, filter_scale,
                                              bias_scale, output_scale) {
      let input_product_scale = input_scale * filter_scale;

      // The following conditions must be guaranteed by the training pipeline.
      OPS_CHECK(Math.abs(input_product_scale - bias_scale) <=
                (1e-6 * Math.min(input_product_scale, bias_scale)));
      OPS_CHECK(input_product_scale >= 0);
      OPS_CHECK(input_product_scale < output_scale);
      let multiplier = input_product_scale / output_scale;
      return multiplier;
    }

    // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#171
    function QuantizeMultiplierSmallerThanOne(double_multiplier) {
      let quantized_multiplier, right_shift, q;
      OPS_CHECK(double_multiplier >= 0.);
      OPS_CHECK(double_multiplier < 1.);
      if (double_multiplier === 0.) {
        quantized_multiplier = 0;
        right_shift = 0;
        return [quantized_multiplier, right_shift];
      }
      OPS_CHECK(double_multiplier > 0.);
      [q, right_shift] = frexp(double_multiplier);
      right_shift *= -1;
      quantized_multiplier = Math.round(q * -(1 << 31));
      OPS_CHECK(quantized_multiplier <= -(1 << 31));
      if (quantized_multiplier == -(1 << 31)) {
        quantized_multiplier /= 2;
        --right_shift;
      }
      OPS_CHECK(right_shift >= 0);
      OPS_CHECK(quantized_multiplier <= nn_ops.INT32_MAX);

      return [quantized_multiplier, right_shift];
    }

    // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#196
    function QuantizeMultiplierGreaterThanOne(double_multiplier) {
      let quantized_multiplier, left_shift, q;
      OPS_CHECK(double_multiplier > 1.);
      [q, left_shift] = frexp(double_multiplier);
      quantized_multiplier = Math.round(q * -(1 << 31));
      OPS_CHECK(quantized_multiplier <= -(1 << 31));
      if (quantized_multiplier == -(1 << 31)) {
        quantized_multiplier /= 2;
        ++left_shift;
      }
      OPS_CHECK(left_shift >= 0);
      OPS_CHECK(quantized_multiplier <= nn_ops.INT32_MAX);
      return [quantized_multiplier, left_shift];
    }

    // reference: https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/OperationsUtils.cpp#281
    function CalculateInputRadius(input_integer_bits, input_left_shift) {
      let max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                               (1 << (31 - input_integer_bits)) /
                               (1 << input_left_shift);
      // Tighten bound using floor.  Suppose that we could use the exact value.
      // After scaling the difference, the result would be at the maximum.  Thus we
      // must ensure that our value has lower magnitude.
      return Math.floor(max_input_rescaled);
    }

    // reference: http://locutus.io/c/math/frexp/index.html
    function frexp (arg) {
      arg = Number(arg);
      const result = [arg, 0];
      if (arg !== 0 && Number.isFinite(arg)) {
        const absArg = Math.abs(arg)
        // Math.log2 was introduced in ES2015, use it when available
        const log2 = Math.log2 || function log2 (n) { return Math.log(n) * Math.LOG2E }
        let exp = Math.max(-1023, Math.floor(log2(absArg)) + 1)
        let x = absArg * Math.pow(2, -exp)

        // These while loops compensate for rounding errors that sometimes occur because of ECMAScript's Math.log2's undefined precision
        // and also works around the issue of Math.pow(2, -exp) === Infinity when exp <= -1024
        while (x < 0.5) {
          x *= 2
          exp--
        }
        while (x >= 1) {
          x *= 0.5
          exp++
        }

        if (arg < 0) {
          x = -x
        }
        result[0] = x
        result[1] = exp
      }
      return result
    }

    function sameShape(input1, input2) {
      if (input1.type != input2.type ||
        input1.runtimeshape.DimensionsCount() != input2.runtimeshape.DimensionsCount()) {
        return false;
      }
      for (let i = 0; i < input1.runtimeshape.DimensionsCount(); i++) {
        if (input1.runtimeshape.Dims(i) != input2.runtimeshape.Dims(i)) {
          return false;
        }
      }
      return true;
    }

    function OPS_CHECK(option) {
      if (!option) {
        throw new Error(`OPS_CHECK failed`);
      }
      return true;
    }

    switch(op) {
      case OperationCode.WEBNN_SUBGRAPH: {
        const execution = operation.execution;

        // workaround for intel/webml-polyfill#674
        inputs.forEach((tensorId, i) => {
          const operand = this._operands[tensorId];
          const length = product(operand.dimensions);
          const view = this._getTensorDataView(operand.type, operand.value, length);
          execution.setInput(i, view);
        });

        // execute subgraph
        await operation.execution.startCompute();
      } break;
      case OperationCode.ADD: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = operands[inputs[2]].value[0];
        let out = operands[outputs[0]];

        let input1_multiplier = 0, input2_multiplier = 0, output_multiplier = 0;
        let input1_shift = 0, input2_shift = 0, output_shift = 0;
        let left_shift = 20;
        if (out.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let twice_max_input_scale = 2 * Math.max(in1.scale, in2.scale);
          let real_input1_multiplier = in1.scale / twice_max_input_scale;
          let real_input2_multiplier = in2.scale / twice_max_input_scale;
          let real_output_multiplier = twice_max_input_scale / ((1 << left_shift) * out.scale);

          [input1_multiplier, input1_shift] = QuantizeMultiplierSmallerThanOne(real_input1_multiplier);
          [input2_multiplier, input2_shift] = QuantizeMultiplierSmallerThanOne(real_input2_multiplier);
          [output_multiplier, output_shift] = QuantizeMultiplierSmallerThanOne(real_output_multiplier);
        }

        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, out);

        // Error check
        OPS_CHECK(in1.type === in2.type);
        OPS_CHECK(in1.runtimeshape.DimensionsCount() <= 4 && in2.runtimeshape.DimensionsCount() <= 4);

        // init arithmeticParams
        let arithmeticParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          input1_offset: -in1.zeroPoint || 0,
          input2_offset: -in2.zeroPoint || 0,
          output_offset: out.zeroPoint || 0,
          output_multiplier: output_multiplier,
          output_shift: -output_shift,
          left_shift: left_shift,
          input1_multiplier: input1_multiplier,
          input1_shift: -input1_shift,
          input2_multiplier: input2_multiplier,
          input2_shift: -input2_shift,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }

        let needBroadCast = !sameShape(in1, in2);
        if (needBroadCast) {
          nn_ops.broadCastAddFloat32(arithmeticParams,
                                     in1.runtimeshape, in1.value,
                                     in2.runtimeshape, in2.value,
                                     out.runtimeshape, out.value);
        } else {
          if (out.type === OperandCode.TENSOR_FLOAT32) {
            nn_ops.addFloat32(arithmeticParams,
                              in1.runtimeshape, in1.value,
                              in2.runtimeshape, in2.value,
                              out.runtimeshape, out.value);
          } else if (out.type === OperandCode.TENSOR_QUANT8_ASYMM) {
            nn_ops.addUint8(arithmeticParams,
                            in1.runtimeshape, in1.value,
                            in2.runtimeshape, in2.value,
                            out.runtimeshape, out.value);
          }
        }
      } break;
      case OperationCode.MUL: {
        allParametersPresent(3, 1);
        let in1 = operands[inputs[0]];
        let in2 = operands[inputs[1]];
        let activation = operands[inputs[2]].value[0];
        let out = operands[outputs[0]];

        let output_multiplier = 0, output_shift = 0;
        if (out.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let input_product_scale = in1.scale * in2.scale;
          let real_multiplier = input_product_scale / out.scale;
          [output_multiplier, output_shift] = QuantizeMultiplierSmallerThanOne(real_multiplier);
        }

        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, out);

        // Error check
        OPS_CHECK(in1.type === in2.type);
        OPS_CHECK(in1.runtimeshape.DimensionsCount() <= 4 && in2.runtimeshape.DimensionsCount() <= 4);

        // init arithmeticParams
        let arithmeticParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          input1_offset: -in1.zeroPoint || 0,
          input2_offset: -in2.zeroPoint || 0,
          output_offset: out.zeroPoint || 0,
          output_multiplier: output_multiplier,
          output_shift: -output_shift,
          left_shift: 0,
          input1_multiplier: 0,
          input1_shift: 0,
          input2_multiplier: 0,
          input2_shift: 0,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }

        let needBroadCast = !sameShape(in1, in2);
        if (needBroadCast) {
          nn_ops.broadCastMulFloat32(arithmeticParams,
                                     in1.runtimeshape, in1.value,
                                     in2.runtimeshape, in2.value,
                                     out.runtimeshape, out.value);
        } else {
          nn_ops.mulFloat32(arithmeticParams,
                            in1.runtimeshape, in1.value,
                            in2.runtimeshape, in2.value,
                            out.runtimeshape, out.value);
        }
      } break;
      case OperationCode.CONV_2D:
      case OperationCode.ATROUS_CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error('Invalid parameters number of CONV_2D');
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let dilationWidth, dilationHeight;
        let filterWidth = filter.runtimeshape.Dims(2);
        let filterHeight = filter.runtimeshape.Dims(1);
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          if (op === OperationCode.CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          [paddingLeft, paddingRight] =
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, dilationWidth, paddingCode);
          [paddingTop, paddingBottom] =
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, dilationHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        let outBatch = output.runtimeshape.Dims(0);
        let outHeight = output.runtimeshape.Dims(1);
        let outWidth = output.runtimeshape.Dims(2);
        let inDepth = input.runtimeshape.Dims(3);

        let output_multiplier = 0, output_shift = 0;
        let typedArray = Float32Array;
        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let real_multiplier = GetQuantizedConvolutionMultipler(input.scale, filter.scale,
                                                                 bias.scale, output.scale);
          [output_multiplier, output_shift] = QuantizeMultiplierSmallerThanOne(real_multiplier);
          typedArray = Uint8Array;
        }

        // init im2col operand
        let im2colDepth = filterWidth * filterHeight * inDepth;
        let im2colDims = [outBatch, outHeight, outWidth, im2colDepth];
        let im2colValue = new typedArray(product(im2colDims));
        let operand = {
          type: OperandCode.TENSOR_FLOAT32,
          dimensions: im2colDims,
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: im2colValue
        }
        let im2colShape = this._allocateRuntimeShape(operand);
        let im2colData = this._allocateTensor(operand);

        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, output);

        // Error check
        OPS_CHECK(input.type === filter.type);
        if (input.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          OPS_CHECK(bias.type === OperandCode.TENSOR_INT32);
        } else {
          OPS_CHECK(input.type === bias.type);
        }

        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(filter.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(bias.runtimeshape.DimensionsCount() === 1);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        OPS_CHECK(filter.runtimeshape.Dims(0) === bias.runtimeshape.Dims(0));
        OPS_CHECK(filter.runtimeshape.Dims(3) === input.runtimeshape.Dims(3));

        // init convParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let convParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          dilation_width_factor: dilationWidth,
          dilation_height_factor: dilationHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          input_offset: -input.zeroPoint || 0,
          weights_offset: -filter.zeroPoint || 0,
          output_offset: output.zeroPoint || 0,
          output_multiplier: output_multiplier,
          output_shift: -output_shift,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }

        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.convFloat32(convParams,
                             input.runtimeshape, input.value,
                             filter.runtimeshape, filter.value,
                             bias.runtimeshape, bias.value,
                             output.runtimeshape, output.value,
                             im2colShape, im2colData);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.convUint8(convParams,
                           input.runtimeshape, input.value,
                           filter.runtimeshape, filter.value,
                           bias.runtimeshape, bias.value,
                           output.runtimeshape, output.value,
                           im2colShape, im2colData);
        }
        im2colShape.delete();
        nn_ops._free(im2colData);
      } break;
      case OperationCode.DEPTHWISE_CONV_2D:
      case OperationCode.ATROUS_DEPTHWISE_CONV_2D: {
        let inCount = inputs.length;
        if (inCount !== 8 && inCount !== 11) {
          throw new Error('Invalid parameters number of DEPTHWISE_CONV_2D');
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let filter = operands[inputs[i++]];
        let bias = operands[inputs[i++]];
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let dilationWidth, dilationHeight;
        let depthMultipler;
        let activation;
        if (inCount === 11) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          if (op === OperationCode.DEPTHWISE_CONV_2D) {
            strideWidth = operands[inputs[i++]].value[0];
            strideHeight = operands[inputs[i++]].value[0];
            [dilationWidth, dilationHeight] = [1, 1];
          } else {
            dilationWidth = operands[inputs[i++]].value[0];
            dilationHeight = operands[inputs[i++]].value[0];
            [strideWidth, strideHeight] = [1, 1];
          }
          depthMultipler = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          let filterWidth = filter.runtimeshape.Dims(2);
          let filterHeight = filter.runtimeshape.Dims(1);

          [paddingLeft, paddingRight] =
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, dilationWidth, paddingCode);
          [paddingTop, paddingBottom] =
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, dilationHeight, paddingCode);
        }
        let output = operands[outputs[0]];

        let output_multiplier = 0, output_shift = 0;
        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let real_multiplier = GetQuantizedConvolutionMultipler(input.scale, filter.scale,
                                                                 bias.scale, output.scale);
          [output_multiplier, output_shift] = QuantizeMultiplierSmallerThanOne(real_multiplier);
        }
        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, output);

        // Error check
        OPS_CHECK(input.type === filter.type);
        if (input.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          OPS_CHECK(bias.type === OperandCode.TENSOR_INT32);
        } else {
          OPS_CHECK(input.type === bias.type);
        }

        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(filter.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(bias.runtimeshape.DimensionsCount() === 1);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        OPS_CHECK(filter.runtimeshape.Dims(3) === bias.runtimeshape.Dims(0));

        // init depthwiseParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let depthwiseParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          dilation_width_factor: dilationWidth,
          dilation_height_factor: dilationHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          depth_multiplier: depthMultipler,
          input_offset: -input.zeroPoint || 0,
          weights_offset: -filter.zeroPoint || 0,
          output_offset: output.zeroPoint || 0,
          output_multiplier: output_multiplier,
          output_shift: -output_shift,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }
        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.depthwiseConvFloat32(depthwiseParams,
                                      input.runtimeshape, input.value,
                                      filter.runtimeshape, filter.value,
                                      bias.runtimeshape, bias.value,
                                      output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.depthwiseConvUint8(depthwiseParams,
                                    input.runtimeshape, input.value,
                                    filter.runtimeshape, filter.value,
                                    bias.runtimeshape, bias.value,
                                    output.runtimeshape, output.value);
        }
      } break;
      case OperationCode.AVERAGE_POOL_2D:
      case OperationCode.MAX_POOL_2D: {
        let inCount = inputs.length;
        if (inCount !== 7 && inCount !== 10) {
          throw new Error(`Invalid parameters number of Pooling ${op}`);
        }
        allParametersPresent(inCount, 1);
        let i = 0;
        let input = operands[inputs[i++]];
        let paddingLeft, paddingRight;  // Just use paddingLeft as paddingWidth
        let paddingTop, paddingBottom;  // Just use paddingTop as paddingHeight
        let strideWidth, strideHeight;
        let filterWidth, filterHeight;
        let activation;
        if (inCount === 10) {
          paddingLeft = operands[inputs[i++]].value[0];
          paddingRight = operands[inputs[i++]].value[0];
          paddingTop = operands[inputs[i++]].value[0];
          paddingBottom = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          filterWidth = operands[inputs[i++]].value[0];
          filterHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];
        } else {
          let paddingCode = operands[inputs[i++]].value[0];
          strideWidth = operands[inputs[i++]].value[0];
          strideHeight = operands[inputs[i++]].value[0];
          filterWidth = operands[inputs[i++]].value[0];
          filterHeight = operands[inputs[i++]].value[0];
          activation = operands[inputs[i++]].value[0];

          let inputWidth = input.runtimeshape.Dims(2);
          let inputHeight = input.runtimeshape.Dims(1);
          [paddingLeft, paddingRight] =
            calculateExplicitPadding(inputWidth, strideWidth, filterWidth, 1, paddingCode);
          [paddingTop, paddingBottom] =
            calculateExplicitPadding(inputHeight, strideHeight, filterHeight, 1, paddingCode);
        }
        let output = operands[outputs[0]];

        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, output);

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() === 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === 4);

        // init poolParams
        let PaddingValues = {
          width: paddingLeft,
          height: paddingTop
        }
        let poolParams = {
          padding_values: PaddingValues,
          stride_width: strideWidth,
          stride_height: strideHeight,
          filter_width: filterWidth,
          filter_height: filterHeight,
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }

        if (op === OperationCode.AVERAGE_POOL_2D) {
          if (output.type === OperandCode.TENSOR_FLOAT32) {
            nn_ops.averagePoolFloat32(poolParams,
                                      input.runtimeshape, input.value,
                                      output.runtimeshape, output.value);
          } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
            nn_ops.averagePoolUint8(poolParams,
                                    input.runtimeshape, input.value,
                                    output.runtimeshape, output.value);
          }
        } else if (op === OperationCode.MAX_POOL_2D) {
          if (output.type === OperandCode.TENSOR_FLOAT32) {
            nn_ops.maxPoolFloat32(poolParams,
                                  input.runtimeshape, input.value,
                                  output.runtimeshape, output.value);
          } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
            nn_ops.maxPoolUint8(poolParams,
                                input.runtimeshape, input.value,
                                output.runtimeshape, output.value);
          }
        }
      } break;
      case OperationCode.SOFTMAX: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let beta = operands[inputs[1]].value[0];
        if (beta <= 0.0) {
          throw new Error('beta must be positive for SOFTMAX');
        }
        let output = operands[outputs[0]];

        let inputMultiplier = 0, inputLeftShift = 0, diffMin = 0;
        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          if (output.zeroPoint != 0 || output.scale != 1 / 256) {
            console.error("incorrect scale / offset for output");
          }
          let kScaledDiffIntegerBits = 5;
          let input_beta_real_multiplier =
              Math.min(1.0 * beta * input.scale * (1 << (31 - kScaledDiffIntegerBits)), -(1 << 31) - 1.0);
          [inputMultiplier, inputLeftShift] = QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier);
          diffMin = -CalculateInputRadius(kScaledDiffIntegerBits, inputLeftShift);
        }

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);

        // init softmaxParams
        let softmaxParams = {
          beta: beta,
          input_multiplier: inputMultiplier,
          input_left_shift: inputLeftShift,
          diff_min: diffMin
        }
        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.softmaxFloat32(softmaxParams,
                                input.runtimeshape, input.value,
                                output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.softmaxUint8(softmaxParams,
                              input.runtimeshape, input.value,
                              output.runtimeshape, output.value);
        }
      } break;
      case OperationCode.RESHAPE: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let targetShape = operands[inputs[1]];  // Dont use targetShape since
                                                // outputShape has been set at first
        let output = operands[outputs[0]];

        let inputDims = [];
        let  outputDims = [];
        for (let i = 0; i < input.runtimeshape.DimensionsCount(); ++i) {
          inputDims.push(input.runtimeshape.Dims(i));
        }
        for (let i = 0; i < output.runtimeshape.DimensionsCount(); ++i) {
          outputDims.push(output.runtimeshape.Dims(i));
        }

        // Error check
        let numInputElements = product(inputDims);
        let numOutputElements = product(outputDims);
        OPS_CHECK(numInputElements === numOutputElements);

        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.reshapeFloat32(input.runtimeshape, input.value,
                                output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.reshapeUint8(input.runtimeshape, input.value,
                              output.runtimeshape, output.value);
        }
      } break;
      case OperationCode.CONCATENATION: {
        if (outputs.length < 1 || inputs.length < 2) {
          throw new Error('Invalid inputs or outputs');
        }
        let numInputTensors = inputs.length - 1;
        let axis = operands[inputs[numInputTensors]].value[0];
        let input0 = operands[inputs[0]];
        let num_dimensions = input0.runtimeshape.DimensionsCount();
        let input_type = input0.type;
        if (axis === -1) {
          axis = num_dimensions - 1;
        }
        let output = operands[outputs[0]];
        let inputShapes = new nn_ops.VectorShape;
        let inputValues = new nn_ops.VectorPtr;
        let inputScale = [];
        let inputZeroPint = [];
        for (let i = 0; i < numInputTensors; ++i) {
          let input = operands[inputs[i]];
          inputShapes.push_back(input.runtimeshape);
          inputValues.push_back(input.value);
          inputScale.push(input.scale);
          inputZeroPint.push(input.zeroPoint);
        }

        // Error check
        OPS_CHECK(axis >= 0 && axis < num_dimensions);
        for (let i = 1; i < numInputTensors; ++i) {
          let input = operands[inputs[i]];
          OPS_CHECK(input.runtimeshape.DimensionsCount() === num_dimensions);
          OPS_CHECK(input.type === input_type);
          for (let d = 0; d < num_dimensions; ++d) {
            if (d != axis) {
              OPS_CHECK(input0.runtimeshape.Dims(d) ===
                        input.runtimeshape.Dims(d));
            }
          }
        }

        // init concatenationParams
        let concatenationParams = {
          axis: axis,
          inputs_count: numInputTensors,
          output_scale: output.scale || 0,
          output_zeropoint: output.zeroPoint || 0
        }

        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.concatenationFloat32(concatenationParams, inputShapes, inputValues,
                                      output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.concatenationUint8(concatenationParams, inputShapes, inputValues,
                                    inputScale, inputZeroPint,
                                    output.runtimeshape, output.value);
        }
        inputShapes.delete();
        inputValues.delete();
      } break;
      case OperationCode.FULLY_CONNECTED: {
        allParametersPresent(4, 1);
        let input = operands[inputs[0]];
        let weights = operands[inputs[1]];
        let bias = operands[inputs[2]];
        let activation = operands[inputs[3]].value[0];
        let output = operands[outputs[0]];

        let output_multiplier = 0, output_shift = 0;
        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let real_multiplier = GetQuantizedConvolutionMultipler(input.scale, weights.scale,
                                                                bias.scale, output.scale);
          [output_multiplier, output_shift] = QuantizeMultiplier(real_multiplier);
        }

        let [float_activation_min, float_activation_max,
             quantized_activation_min, quantized_activation_max] = calculateActivationRange(activation, output);

        // Error check
        OPS_CHECK(weights.runtimeshape.DimensionsCount() === 2);

        // init fullyConnectedParams
        let fullyConnectedParams = {
          float_activation_min: float_activation_min,
          float_activation_max: float_activation_max,
          input_offset: -input.zeroPoint || 0,
          weights_offset: -weights.zeroPoint || 0,
          output_offset: output.zeroPoint || 0,
          output_multiplier: output_multiplier,
          output_shift: output_shift,
          quantized_activation_min: quantized_activation_min,
          quantized_activation_max: quantized_activation_max
        }

        if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.fullyConnectedFloat32(fullyConnectedParams,
                                       input.runtimeshape, input.value,
                                       weights.runtimeshape, weights.value,
                                       bias.runtimeshape, bias.value,
                                       output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          nn_ops.fullyConnectedUint8(fullyConnectedParams,
                                     input.runtimeshape, input.value,
                                     weights.runtimeshape, weights.value,
                                     bias.runtimeshape, bias.value,
                                     output.runtimeshape, output.value);
        }
      } break;
      case OperationCode.RESIZE_BILINEAR: {
        let inCount = inputs.length;
        if (inCount !== 3 && inCount !== 4) {
          throw new Error(`Invalid parameters number of resize bilinear ${op}`);
        }
        allParametersPresent(inCount, 1);
        let input = operands[inputs[0]];
        let newHeight = operands[inputs[1]].value[0]; // Dont use newHeight and newWidth
        let newWidth = operands[inputs[2]].value[0];  // since outputShape has been set at first
        // init resizeBilinearParams
        // default set align_corners to false
        let resizeBilinearParams = {
          align_corners: false
        };
        if (inCount === 4) {
          resizeBilinearParams.align_corners =
              operands[inputs[3]].value[0] !== 0;
        }
        let output = operands[outputs[0]];
        let outSizeHeight = output.runtimeshape.Dims(1);
        let outSizeWidth = output.runtimeshape.Dims(2);

        let outSizeDims = [2];
        let outSizeValue = new Int32Array([outSizeHeight, outSizeWidth]);
        let operand = {
          type: OperandCode.TENSOR_INT32,
          dimensions: outSizeDims,
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: outSizeValue
        }
        let outSizeShape = this._allocateRuntimeShape(operand);
        let outSizeData = this._allocateTensor(operand);

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() <= 4);

        nn_ops.resizeBilinearFloat32(resizeBilinearParams,
                                     input.runtimeshape, input.value,
                                     outSizeShape, outSizeData,
                                     output.runtimeshape, output.value);
        outSizeShape.delete();
        nn_ops._free(outSizeData);
      } break;
      case OperationCode.TANH: {
        allParametersPresent(1, 1);
        let input = operands[inputs[0]];
        let output = operands[outputs[0]];

        nn_ops.tanhFloat32(input.runtimeshape, input.value,
                           output.runtimeshape, output.value);
      } break;
      case OperationCode.MAXIMUM: {
        allParametersPresent(2, 1);
        let input1 = operands[inputs[0]];
        let input2 = operands[inputs[1]];
        let output = operands[outputs[0]];

        // Error check
        OPS_CHECK(input1.type === input2.type);

        nn_ops.maximumFloat32(input1.runtimeshape, input1.value,
                              input2.runtimeshape, input2.value,
                              output.runtimeshape, output.value);
      } break;
      case OperationCode.BATCH_TO_SPACE_ND: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let blockShape = operands[inputs[1]];
        let output = operands[outputs[0]];

        // set a default crops
        let operand = {
          type: OperandCode.TENSOR_INT32,
          dimensions: [2, 2],
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: [0, 0, 0, 0]
        };
        let cropsShape = this._allocateRuntimeShape(operand);
        let cropsData = this._allocateTensor(operand);

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() <= 4);

        nn_ops.batchToSpaceNDFloat32(input.runtimeshape, input.value,
                                     blockShape.runtimeshape, blockShape.value,
                                     cropsShape, cropsData,
                                     output.runtimeshape, output.value);
        cropsShape.delete();
        nn_ops._free(cropsData);
      } break;
      case OperationCode.TRANSPOSE: {
        let inCount = inputs.length;
        if (inCount !== 1 && inCount !== 2) {
          throw new Error('Invalid parameters number of TRANSPOSE');
        }
        allParametersPresent(inCount, 1);
        let input = operands[inputs[0]];
        let perm = [];
        if (inCount === 1) {
          // set a default perm
          let n = input.runtimeshape.DimensionsCount();
          for (let i = 0; i < n; ++i) {
            perm[i] = n - i;
          }
        } else {
          perm = modelOperands[inputs[1]].value;
        }
        let output = operands[outputs[0]];

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() <= 4);
        OPS_CHECK(output.runtimeshape.DimensionsCount() === perm.length);

        // init transposeParams
        let transposeParams = {
          perm: perm,
          perm_count: perm.length
        }

        nn_ops.transposeFloat32(transposeParams,
                                input.runtimeshape, input.value,
                                output.runtimeshape, output.value);
      } break;
      case OperationCode.ARGMAX: {
        allParametersPresent(2, 1);
        let input1 = operands[inputs[0]];
        let input2 = operands[inputs[1]];
        let operand = {
          type: OperandCode.TENSOR_INT32,
          dimensions: [1],
          numberOfConsumers: 0,
          lifetime: OperandLifetime.CONSTANT_REFERENCE,
          value: [input2.value[0]]
        };
        let axisData = this._allocateTensor(operand);
        let output = operands[outputs[0]];

        nn_ops.argMaxFloat32(input1.runtimeshape, input1.value,
                             axisData, output.runtimeshape, output.value);
        nn_ops._free(axisData);
      } break;
      case OperationCode.LOGISTIC: {
        allParametersPresent(1, 1);
        let input = operands[inputs[0]];
        let output = operands[outputs[0]];

        // Error check
        OPS_CHECK(input.runtimeshape.DimensionsCount() <= 4);

        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          if (output.zeroPoint != 0 || output.scale != 1 / 256) {
            console.error("incorrect scale / offset for output");
          };
          let input_zero_point = input.zeroPoint;
          let input_range_radius = 0;
          let input_multiplier = 0;
          let input_left_shift= 0;

          let kInputIntegerBits = 4;
          let input_real_multiplier = input.scale * (1 << (31 - kInputIntegerBits));
          [input_multiplier, input_left_shift] = QuantizeMultiplierGreaterThanOne(input_real_multiplier);
          input_range_radius = CalculateInputRadius(kInputIntegerBits, input_left_shift);

          let logisticParams = {
            // uint8 inference params.
            input_zero_point: input_zero_point,
            input_range_radius: input_range_radius,
            input_multiplier: input_multiplier,
            input_left_shift: input_left_shift
          };

          nn_ops.logisticUint8(logisticParams,
                               input.runtimeshape, input.value,
                               output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.logisticFloat32(input.runtimeshape, input.value,
                                 output.runtimeshape, output.value);
        };
      } break;
      case OperationCode.PRELU: {
        allParametersPresent(2, 1);
        let input = operands[inputs[0]];
        let alpha = operands[inputs[1]];
        let output = operands[outputs[0]];

        if (output.type === OperandCode.TENSOR_QUANT8_ASYMM) {
          let input_offset = -input.zeroPoint || 0;
          let alpha_offset = -alpha.zeroPoint || 0;
          let output_offset = output.zeroPoint || 0;

          let input_product_scale = input.scale * alpha.scale;
          let real_multiplier = input_product_scale / output.scale;
          let [output_multiplier, output_shift] = QuantizeMultiplier(real_multiplier);

          let preluParams = {
            input_offset: input_offset,
            alpha_offset: alpha_offset,
            output_offset: output_offset,
            output_multiplier: output_multiplier,
            output_shift: output_shift
          };

          nn_ops.preluUint8(preluParams, 
                            input.runtimeshape, input.value,
                            alpha.runtimeshape, alpha.value,
                            output.runtimeshape, output.value);
        } else if (output.type === OperandCode.TENSOR_FLOAT32) {
          nn_ops.preluFloat32(input.runtimeshape, input.value,
                              alpha.runtimeshape, alpha.value,
                              output.runtimeshape, output.value);
        };
      } break;
      default: {
        throw new Error(`Operation ${op} is not supported`);
      }
    }
  }

  _setTensorData(type, ptr, data) {
    const nn_ops = this._nn_ops;
    if (type === OperandCode.TENSOR_FLOAT32) {
      nn_ops.HEAPF32.set(data, ptr >> 2);
    } else if (type === OperandCode.TENSOR_INT32) {
      nn_ops.HEAP32.set(data, ptr >> 2);
    } else if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
      nn_ops.HEAPU8.set(data, ptr);
    } else {
      throw new Error(`Operand type ${type} is not supported`);
    }
  }

  _getTensorData(type, ptr, buffer) {
    const nn_ops = this._nn_ops;
    let view;
    if (type === OperandCode.TENSOR_FLOAT32) {
      view = new Float32Array(nn_ops.HEAPF32.buffer, ptr, buffer.length);
    } else if (type === OperandCode.TENSOR_INT32) {
      view = new Int32Array(nn_ops.HEAP32.buffer, ptr, buffer.length);
    } else if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
      view = new Uint8Array(nn_ops.HEAPU8.buffer, ptr, buffer.length);
    } else {
      throw new Error(`Operand type ${type} is not supported`);
    }
    buffer.set(view);
  }

  _getTensorDataView(type, ptr, length) {
    const nn_ops = this._nn_ops;
    let view;
    if (type === OperandCode.TENSOR_FLOAT32) {
      view = new Float32Array(nn_ops.HEAPF32.buffer, ptr, length);
    } else if (type === OperandCode.TENSOR_INT32) {
      view = new Int32Array(nn_ops.HEAP32.buffer, ptr, length);
    } else if (type === OperandCode.TENSOR_QUANT8_ASYMM) {
      view = new Uint8Array(nn_ops.HEAPU8.buffer, ptr, length);
    } else {
      throw new Error(`Operand type ${type} is not supported`);
    }
    return view;
  }


  _allocateTensor(operand) {
    const nn_ops = this._nn_ops;
    let byteLength = utils.sizeOfTensorData(operand.type, operand.dimensions);
    let ptr = nn_ops._malloc(byteLength);
    if (operand.lifetime === OperandLifetime.CONSTANT_REFERENCE) {
      this._setTensorData(operand.type, ptr, operand.value);
    }
    return ptr;
  }

  _allocateRuntimeShape(operand) {
    const nn_ops = this._nn_ops;
    let RuntimeShape = new nn_ops.RuntimeShape(operand.dimensions.length);
    for (let i = 0; i < RuntimeShape.DimensionsCount(); ++i) {
      RuntimeShape.SetDim(i, operand.dimensions[i]);
    }
    return RuntimeShape;
  }

  _deleteAll() {
    this._toDelete.tensorValue.forEach(tensorValue => {
      this._nn_ops._free(tensorValue);
    });
    this._toDelete.tensorShape.forEach(tensorShape => {
      tensorShape.delete();
    });
    this._model._operands = [];
  }

  getSubgraphsSummary() {
    return this._subgraphs.map((graph, i) =>
        `Subgraph ${i}\t (${graph.backend}):\t{${graph.summary}}`);
  }

  dumpProfilingResults() {
    const res = this._profiler.flush();
    const timings = [];
    const supportedOps = Array.from(this._supportedOps)
        .map(op => findKey(OperationCode, op));
    const mode = this._eager ? 'Eager' : 'Graph';

    if (res.epochs <= 0) {
      console.warn(`Report will be available after at least ${warmUpRuns + 1} executions.`);
    } else {
      for (const [i, op] of this._operations.entries()) {
        const opTime = res.elpased[i];
        if (op.type !== OperationCode.WEBNN_SUBGRAPH) {
          timings.push({
            backend: 'WASM',
            summary: findKey(OperationCode, op.type) + ' x 1',
            elpased: opTime
          });
        } else {
          timings.push({
            backend: 'WebNN',
            summary: op.summary,
            elpased: opTime
          });
        }
      }
    }

    return {
      mode: mode,
      warmUpRuns: warmUpRuns,
      epochs: res.epochs,
      supportedOps: supportedOps,
      timings: timings
    };
  }
}
