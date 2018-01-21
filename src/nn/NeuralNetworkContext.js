import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'
import Model from './Model'
import Compilation from './Compilation'
import Execution from './Execution'

export default class NeuralNetworkContext {
  constructor() {
    this._initOperandTypes();
    this._initOperationTypes();
    this._initFusedActivationFunctionTypes();
    this._initImplicitPaddingTypes();
    this._initExecutionPreferenceTypes();
  }

  /**
   * Create a model object.
   * 
   * @param {string} name - The model name.
   */
  createModel(name) {
    return new Model(name);
  }

   /**
   * Create a compilation from model.
   * 
   * @param {Model} model - the model object.
   * @returns {Compilation} - the compilation object.
   */
  createCompilation(model) {
    if (!model._completed) {
      throw new Error('Model is not finished');
    }
    return new Compilation(model);
  }

  /**
   * Create a executino from compilation.
   * 
   * @param {Compilation} compilation - the compilation object.
   * @returns {Execution} - the execution object.
   */
  createExecution(compilation) {
    if (!compilation._finished) {
      throw new Error('Compilation is not finished');
    }
    return new Execution(compilation);
  }

  _initOperandTypes() {
    this.FLOAT32 = OperandCode.FLOAT32;
    this.INT32 = OperandCode.INT32;
    this.UINT32 = OperandCode.UINT32;
    this.TENSOR_FLOAT32 = OperandCode.TENSOR_FLOAT32;
    this.TENSOR_INT32 = OperandCode.TENSOR_INT32;
    this.TENSOR_QUANT8_ASYMM = OperandCode.TENSOR_QUANT8_ASYMM;
  }

  _initOperationTypes() {
    this.ADD = OperationCode.ADD;
    this.AVERAGE_POOL_2D = OperationCode.AVERAGE_POOL_2D;
    this.CONCATENATION = OperationCode.CONCATENATION;
    this.CONV_2D = OperationCode.CONV_2D;
    this.DEPTHWISE_CONV_2D = OperationCode.DEPTHWISE_CONV_2D;
    this.DEPTH_TO_SPACE = OperationCode.DEPTH_TO_SPACE;
    this.DEQUANTIZE = OperationCode.DEQUANTIZE;
    this.EMBEDDING_LOOKUP = OperationCode.EMBEDDING_LOOKUP;
    this.FLOOR = OperationCode.FLOOR;
    this.FULLY_CONNECTED = OperationCode.FULLY_CONNECTED;
    this.HASHTABLE_LOOKUP = OperationCode.HASHTABLE_LOOKUP;
    this.L2_NORMALIZATION = OperationCode.L2_NORMALIZATION;
    this.L2_POOL_2D = OperationCode.L2_POOL_2D;
    this.LOCAL_RESPONSE_NORMALIZATION = OperationCode.LOCAL_RESPONSE_NORMALIZATION;
    this.LOGISTIC = OperationCode.LOGISTIC;
    this.LSH_PROJECTION = OperationCode.LSH_PROJECTION;
    this.LSTM = OperationCode.LSTM;
    this.MAX_POOL_2D = OperationCode.MAX_POOL_2D;
    this.MUL = OperationCode.MUL;
    this.RELU = OperationCode.RELU;
    this.RELU1 = OperationCode.RELU1;
    this.RELU6 = OperationCode.RELU6;
    this.RESHAPE = OperationCode.RESHAPE;
    this.RESIZE_BILINEAR = OperationCode.RESIZE_BILINEAR;
    this.RNN = OperationCode.RNN;
    this.SOFTMAX = OperationCode.SOFTMAX;
    this.SPACE_TO_DEPTH = OperationCode.SPACE_TO_DEPTH;
    this.SVDF = OperationCode.SVDF;
    this.TANH = OperationCode.TANH;
  }

  _initFusedActivationFunctionTypes() {
    this.FUSED_NONE = FuseCode.NONE;
    this.FUSED_RELU = FuseCode.RELU;
    this.FUSED_RELU1 = FuseCode.RELU1;
    this.FUSED_RELU6 = FuseCode.RELU6;
  }

  _initImplicitPaddingTypes() {
    this.PADDING_SAME = PaddingCode.SAME;
    this.PADDING_VALID = PaddingCode.VALID;
  }

  _initExecutionPreferenceTypes() {
    this.PREFER_LOW_POWER = PreferenceCode.LOW_POWER;
    this.PREFER_FAST_SINGLE_ANSWER = PreferenceCode.FAST_SINGLE_ANSWER;
    this.PREFER_SUSTAINED_SPEED = PreferenceCode.SUSTAINED_SPEED;
  }
}
