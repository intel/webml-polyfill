#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "external/tensorflow/tensorflow/lite/kernels/internal/types.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"

#include <vector>
#include <cmath>
#include <iostream>

using namespace emscripten;

namespace binding_utils {
  struct ReshapeParams {
    uint32_t size_count;
  };

  // Factors set and get function
  val getRuntimeShapeDimensions(const tflite::RuntimeShape& shape) {
    emscripten::val js_dims = emscripten::val::array();
    for (int i = 0; i < shape.DimensionsCount(); i++) {
      js_dims.call<void>("push", shape.Dims(i));
    }
    return js_dims;
  }

  void setRuntimeShapeDimensions(tflite::RuntimeShape& shape, val js_dims) {
    std::vector<int> tmp = vecFromJSArray<int32_t>(js_dims);
    for (int i = 0; i < tmp.size(); i++) {
      shape.SetDim(i, tmp.at(i));
    }
  }

  val getRuntimeShapeSize(const tflite::RuntimeShape& shape) {
    emscripten::val js_val = val(shape.DimensionsCount());
    return js_val;
  }

  void setRuntimeShapeSize(tflite::RuntimeShape& shape, int32_t size) {
    shape.Resize(size);
  }

  template <typename T>
  val getPaddingValues(const T& params) {
    emscripten::val js_dims = emscripten::val::array();
    js_dims.call<void>("push", params.padding_values.width);
    js_dims.call<void>("push", params.padding_values.height);
    return js_dims;
  }

  template <typename T>
  void setPaddingValues(T& params, val js_dims) {
    std::vector<uint32_t> tmp = vecFromJSArray<uint32_t>(js_dims);
    params.padding_values.width = tmp.at(0);
    params.padding_values.height = tmp.at(1);
  }

  template <typename T>
  val getDilationFactor(const T& params) {
    emscripten::val js_dims = emscripten::val::array();
    js_dims.call<void>("push", params.dilation_width_factor);
    js_dims.call<void>("push", params.dilation_height_factor);
    return js_dims;
  }

  template <typename T>
  void setDilationFactor(T& params, val js_dims) {
    std::vector<int32_t> tmp = vecFromJSArray<int32_t>(js_dims);
    params.dilation_width_factor = tmp.at(0);
    params.dilation_height_factor = tmp.at(1);
  }

  template <typename T>
  val getStrides(const T& params) {
    emscripten::val js_dims = emscripten::val::array();
    js_dims.call<void>("push", params.stride_width);
    js_dims.call<void>("push", params.stride_height);
    return js_dims;
  }

  template <typename T>
  void setStrides(T& params, val js_dims) {
    std::vector<uint32_t> tmp = vecFromJSArray<uint32_t>(js_dims);
    params.stride_width = tmp.at(0);
    params.stride_height = tmp.at(1);
  }

  template <typename T>
  val getActivationRange(const T& params) {
    emscripten::val js_dims = emscripten::val::array();
    js_dims.call<void>("push", params.float_activation_min);
    js_dims.call<void>("push", params.float_activation_max);
    return js_dims;
  }

  template <typename T>
  void setActivationRange(T& params, val js_dims) {
    std::vector<float> tmp = vecFromJSArray<float>(js_dims);
    params.float_activation_min = tmp.at(0);
    params.float_activation_max = tmp.at(1);
  }

  template <typename T>
  val getDepthMultiplier(const T& params) {
    emscripten::val js_val = val(params.depth_multiplier);
    return js_val;
  }

  template <typename T>
  void setDepthMultiplier(T& params, int32_t depth_multiplier) {
    params.depth_multiplier = depth_multiplier;
  }

  template <typename T>
  val getBeta(const T& params) {
    emscripten::val js_val = val(params.beta);
    return js_val;
  }

  template <typename T>
  void setBeta(T& params, float beta) {
    params.beta = beta;
  }

  template <typename T>
  val getFilters(const T& params) {
    emscripten::val js_dims = emscripten::val::array();
    js_dims.call<void>("push", params.filter_width);
    js_dims.call<void>("push", params.filter_height);
    return js_dims;
  }

  template <typename T>
  void setFilters(T& params, val js_dims) {
    std::vector<int32_t> tmp = vecFromJSArray<int32_t>(js_dims);
    params.filter_width = tmp.at(0);
    params.filter_height = tmp.at(1);
  }

  template <typename T>
  val getAlignCorners(const T& params) {
    emscripten::val js_val = val(params.align_corners);
    return js_val;
  }

  template <typename T>
  void setAlignCorners(T& params, bool align_corners) {
    params.align_corners = align_corners;
  }

  template <typename T>
  val getAxis(const T& params) {
    emscripten::val js_val = val(params.axis);
    return js_val;
  }

  template <typename T>
  void setAxis(T& params, int32_t axis) {
    params.axis = axis;
  }

  template <typename T>
  val getInputsCount(const T& params) {
    emscripten::val js_val = val(params.inputs_count);
    return js_val;
  }

  template <typename T>
  void setInputsCount(T& params, int32_t inputs_count) {
    params.inputs_count = inputs_count;
  }

  template <typename T>
  val getSizeCount(const T& params) {
    emscripten::val js_val = val(params.size_count);
    return js_val;
  }

  template <typename T>
  void setSizeCount(T& params, uint32_t size_count) {
    params.size_count = size_count;
  }

  // Operation wrappers.
  bool addFloat32Wrapper(const tflite::ArithmeticParams& op_params,
                                  const tflite::RuntimeShape& input1_shape, const intptr_t input1_data, 
                                  const tflite::RuntimeShape& input2_shape, const intptr_t input2_data, 
                                  const tflite::RuntimeShape& output_shape, intptr_t output_data) {
    tflite::optimized_ops::Add(op_params,
                               input1_shape, (const float*) input1_data,
                               input2_shape, (const float*) input2_data,
                               output_shape, (float*) output_data);
    return true;
  }

  bool broadCastAddFloat32Wrapper(const tflite::ArithmeticParams& op_params,
                                  const tflite::RuntimeShape& input1_shape, const intptr_t input1_data, 
                                  const tflite::RuntimeShape& input2_shape, const intptr_t input2_data, 
                                  const tflite::RuntimeShape& output_shape, intptr_t output_data) {
    tflite::optimized_ops::BroadcastAdd4DSlow(op_params,
                                              input1_shape, (const float*) input1_data,
                                              input2_shape, (const float*) input2_data,
                                              output_shape, (float*) output_data);
    return true;
  }

  bool mulFloat32Wrapper(const tflite::ArithmeticParams& op_params,
                         const tflite::RuntimeShape& input1_shape, const intptr_t input1_data, 
                         const tflite::RuntimeShape& input2_shape, const intptr_t input2_data, 
                         const tflite::RuntimeShape& output_shape, intptr_t output_data) {
    tflite::optimized_ops::Mul(op_params,
                               input1_shape, (const float*) input1_data,
                               input2_shape, (const float*) input2_data,
                               output_shape, (float*) output_data);
    return true;
  }

  bool broadCastMulFloat32Wrapper(const tflite::ArithmeticParams& op_params,
                                  const tflite::RuntimeShape& input1_shape, const intptr_t input1_data, 
                                  const tflite::RuntimeShape& input2_shape, const intptr_t input2_data, 
                                  const tflite::RuntimeShape& output_shape, intptr_t output_data) {
    tflite::optimized_ops::BroadcastMul4DSlow(op_params,
                                              input1_shape, (const float*) input1_data,
                                              input2_shape, (const float*) input2_data,
                                              output_shape, (float*) output_data);
    return true;
  }

  bool floorFloat32Wrapper(const tflite::RuntimeShape& input_shape, const intptr_t inputData, 
                           const tflite::RuntimeShape& output_shape, intptr_t outputData) {
    tflite::optimized_ops::Floor(input_shape, (const float*)inputData,
                                 output_shape, (float*)outputData);
    return true;
  }

  bool depthwiseConvFloat32Wrapper(const tflite::DepthwiseParams& op_params,
                                   const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                                   const tflite::RuntimeShape& filterShape, const intptr_t filterData, 
                                   const tflite::RuntimeShape& biasShape, const intptr_t biasData, 
                                   const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::DepthwiseConv(op_params,
                                         inputShape, (const float*)inputData, 
                                         filterShape, (const float*)filterData, 
                                         biasShape, (const float*)biasData, 
                                         outputShape, (float*)outputData);
    return true;
  }

  bool convFloat32Wrapper(const tflite::ConvParams& op_params, 
                          const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                          const tflite::RuntimeShape& filterShape, const intptr_t filterData, 
                          const tflite::RuntimeShape& biasShape, const intptr_t biasData, 
                          const tflite::RuntimeShape& outputShape, intptr_t outputData,
                          const tflite::RuntimeShape& im2colShape, intptr_t im2colData) {
    tflite::optimized_ops::Conv(op_params, 
                                inputShape, (const float*)inputData, 
                                filterShape, (const float*)filterData, 
                                biasShape, (const float*)biasData, 
                                outputShape, (float*)outputData, 
                                im2colShape, (float*)im2colData);
    return true;
  }

  bool averagePoolFloat32Wrapper(const tflite::PoolParams op_params,
                                 const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                                 const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::AveragePool(op_params,
                                       inputShape, (const float*)inputData,
                                       outputShape, (float*)outputData);
    return true;
  }

  bool maxPoolFloat32Wrapper(const tflite::PoolParams op_params,
                             const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                             const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::MaxPool(op_params,
                                   inputShape, (const float*)inputData,
                                   outputShape, (float*)outputData);
    return true;
  }

  bool softmaxFloat32Wrapper(const tflite::SoftmaxParams op_params,
                             const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                             const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::Softmax(op_params, inputShape, (const float*)inputData,
                                   outputShape, (float*)outputData);
    return true;
  }

  bool reshapeGenericWrapper(const binding_utils::ReshapeParams op_params,
                             const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                             const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    // implement it by self due to no reshape op in tflite::optimized_ops
    uint32_t size_count = (uint32_t)op_params.size_count;
    memcpy((void*)outputData, (const void*)inputData, size_count);
    return true;
  }

  bool concatenationFloat32Wrapper(const tflite::ConcatenationParams op_params,  
                                   const std::vector<tflite::RuntimeShape*> inputShapes, 
                                   const std::vector<intptr_t>& inputDataPtrs,
                                   const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::Concatenation<float>(op_params,
                                                inputShapes.data(),
                                                ((const std::vector<const float*>&)inputDataPtrs).data(), 
                                                outputShape, (float*)outputData);
    
    return true;
  }

  bool fullyConnectedFloat32Wrapper(const tflite::FullyConnectedParams op_params,
                                    const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                                    const tflite::RuntimeShape& weightsShape, const intptr_t weightsData, 
                                    const tflite::RuntimeShape& biasShape, const intptr_t biasData, 
                                    const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::FullyConnected(op_params, 
                                          inputShape, (const float*)inputData, 
                                          weightsShape, (const float*)weightsData, 
                                          biasShape, (const float*)biasData,
                                          outputShape, (float*)outputData);
    return true;
  }

  bool resizeBilinearFloat32Wrapper(const tflite::ResizeBilinearParams op_params,
                                    const tflite::RuntimeShape& inputShape, const intptr_t inputData, 
                                    const tflite::RuntimeShape& outSizeShape, const intptr_t outSizeData,
                                    const tflite::RuntimeShape& outputShape, intptr_t outputData) {
    tflite::optimized_ops::ResizeBilinear(op_params, 
                                          inputShape, (const float*)inputData, 
                                          outSizeShape, (const int32_t*)outSizeData, 
                                          outputShape, (float*)outputData);
    return true;
  }

}

EMSCRIPTEN_BINDINGS(nn)
{
  constant("NONE", (int32_t)FusedActivationFunc::NONE);
  constant("RELU", (int32_t)FusedActivationFunc::RELU);
  constant("RELU1", (int32_t)FusedActivationFunc::RELU1);
  constant("RELU6", (int32_t)FusedActivationFunc::RELU6);

  constant("FLOAT32", (int32_t)OperandType::FLOAT32);
  constant("INT32", (int32_t)OperandType::INT32);
  constant("UINT32", (int32_t)OperandType::UINT32);
  constant("TENSOR_FLOAT32", (int32_t)OperandType::TENSOR_FLOAT32);
  constant("TENSOR_INT32", (int32_t)OperandType::TENSOR_INT32);
  constant("TENSOR_QUANT8_ASYMM", (int32_t)OperandType::TENSOR_QUANT8_ASYMM);

  constant("MAX", std::numeric_limits<float>::max());
  constant("LOWEST", std::numeric_limits<float>::lowest());
  constant("MIN", std::numeric_limits<float>::min());

  class_<tflite::RuntimeShape>("RuntimeShape")
    .constructor<int>(select_overload<tflite::RuntimeShape(int)>([](int dimensions_count) {
        return tflite::RuntimeShape(dimensions_count);
      }
    ))
    .property("dims", &binding_utils::getRuntimeShapeDimensions, &binding_utils::setRuntimeShapeDimensions)
    .property("size", &binding_utils::getRuntimeShapeSize, &binding_utils::setRuntimeShapeSize)
    ;

  class_<tflite::ConvParams>("ConvParams")
    .constructor<>()
    .property("padding_values", &binding_utils::getPaddingValues<tflite::ConvParams>, &binding_utils::setPaddingValues<tflite::ConvParams>)
    .property("strides", &binding_utils::getStrides<tflite::ConvParams>, &binding_utils::setStrides<tflite::ConvParams>)
    .property("dilation_factors", &binding_utils::getDilationFactor<tflite::ConvParams>, &binding_utils::setDilationFactor<tflite::ConvParams>)
    .property("float_activation_range", &binding_utils::getActivationRange<tflite::ConvParams>, &binding_utils::setActivationRange<tflite::ConvParams>)
    ;

  class_<tflite::DepthwiseParams>("DepthwiseParams")
    .constructor<>()
    .property("padding_values", &binding_utils::getPaddingValues<tflite::DepthwiseParams>, &binding_utils::setPaddingValues<tflite::DepthwiseParams>)
    .property("strides", &binding_utils::getStrides<tflite::DepthwiseParams>, &binding_utils::setStrides<tflite::DepthwiseParams>)
    .property("dilation_factors", &binding_utils::getDilationFactor<tflite::DepthwiseParams>, &binding_utils::setDilationFactor<tflite::DepthwiseParams>)
    .property("float_activation_range", &binding_utils::getActivationRange<tflite::DepthwiseParams>, &binding_utils::setActivationRange<tflite::DepthwiseParams>)
    .property("depth_multiplier", &binding_utils::getDepthMultiplier<tflite::DepthwiseParams>, &binding_utils::setDepthMultiplier<tflite::DepthwiseParams>)
    ;

  class_<tflite::SoftmaxParams>("SoftmaxParams")
    .constructor<>()
    .property("beta", &binding_utils::getBeta<tflite::SoftmaxParams>, &binding_utils::setBeta<tflite::SoftmaxParams>)
    ;

  class_<tflite::PoolParams>("PoolParams")
    .constructor<>()
    .property("padding_values", &binding_utils::getPaddingValues<tflite::PoolParams>, &binding_utils::setPaddingValues<tflite::PoolParams>)
    .property("strides", &binding_utils::getStrides<tflite::PoolParams>, &binding_utils::setStrides<tflite::PoolParams>)
    .property("filters", &binding_utils::getFilters<tflite::PoolParams>, &binding_utils::setFilters<tflite::PoolParams>)
    .property("float_activation_range", &binding_utils::getActivationRange<tflite::PoolParams>, &binding_utils::setActivationRange<tflite::PoolParams>)
    ;

  class_<tflite::ResizeBilinearParams>("ResizeBilinearParams")
    .constructor<>()
    .property("align_corners", &binding_utils::getAlignCorners<tflite::ResizeBilinearParams>, &binding_utils::setAlignCorners<tflite::ResizeBilinearParams>)
    ;

  class_<tflite::ConcatenationParams>("ConcatenationParams")
    .constructor<>()
    .property("axis", &binding_utils::getAxis<tflite::ConcatenationParams>, &binding_utils::setAxis<tflite::ConcatenationParams>)
    .property("inputs_count", &binding_utils::getInputsCount<tflite::ConcatenationParams>, &binding_utils::setInputsCount<tflite::ConcatenationParams>)
    ;

  class_<tflite::FullyConnectedParams>("FullyConnectedParams")
    .constructor<>()
    .property("float_activation_range", &binding_utils::getActivationRange<tflite::FullyConnectedParams>, &binding_utils::setActivationRange<tflite::FullyConnectedParams>)
    ;

  class_<tflite::ArithmeticParams>("ArithmeticParams")
    .constructor<>()
    .property("float_activation_range", &binding_utils::getActivationRange<tflite::ArithmeticParams>, &binding_utils::setActivationRange<tflite::ArithmeticParams>)
    ;

  class_<binding_utils::ReshapeParams>("ReshapeParams")
    .constructor<>()
    .property("size_count", &binding_utils::getSizeCount<binding_utils::ReshapeParams>, &binding_utils::setSizeCount<binding_utils::ReshapeParams>)
    ;

  register_vector<tflite::RuntimeShape*>("VectorShape");
  register_vector<intptr_t>("VectorPtr");


  // Operations.
  function("addFloat32", &binding_utils::addFloat32Wrapper, allow_raw_pointers());
  function("broadCastAddFloat32", &binding_utils::broadCastAddFloat32Wrapper, allow_raw_pointers());
  function("mulFloat32", &binding_utils::mulFloat32Wrapper, allow_raw_pointers());
  function("broadCastMulFloat32", &binding_utils::broadCastMulFloat32Wrapper, allow_raw_pointers());
  function("floorFloat32", &binding_utils::floorFloat32Wrapper, allow_raw_pointers());
  function("depthwiseConvFloat32", &binding_utils::depthwiseConvFloat32Wrapper, allow_raw_pointers());
  function("convFloat32", &binding_utils::convFloat32Wrapper, allow_raw_pointers());
  function("averagePoolFloat32", &binding_utils::averagePoolFloat32Wrapper, allow_raw_pointers());
  function("softmaxFloat32", &binding_utils::softmaxFloat32Wrapper, allow_raw_pointers());
  function("reshapeGeneric", &binding_utils::reshapeGenericWrapper, allow_raw_pointers());
  function("maxPoolFloat32", &binding_utils::maxPoolFloat32Wrapper, allow_raw_pointers());
  function("concatenationFloat32", &binding_utils::concatenationFloat32Wrapper, allow_raw_pointers());
  function("fullyConnectedFloat32", &binding_utils::fullyConnectedFloat32Wrapper, allow_raw_pointers());
  function("resizeBilinearFloat32", &binding_utils::resizeBilinearFloat32Wrapper, allow_raw_pointers());

  // TODO: operation wrappers
  /*
  function("l2PoolFloat32", &binding_utils::l2PoolFloat32Wrapper, allow_raw_pointers());
  function("maxPoolQuant8", &binding_utils::maxPoolQuant8Wrapper, allow_raw_pointers());
  function("reluFloat32", &binding_utils::reluFloat32Wrapper, allow_raw_pointers());
  function("relu1Float32", &binding_utils::relu1Float32Wrapper, allow_raw_pointers());
  function("relu6Float32", &binding_utils::relu6Float32Wrapper, allow_raw_pointers());
  function("tanhFloat32", &binding_utils::tanhFloat32Wrapper, allow_raw_pointers());
  function("logisticFloat32", &binding_utils::logisticFloat32Wrapper, allow_raw_pointers());
  function("reluQuant8", &binding_utils::reluQuant8Wrapper, allow_raw_pointers());
  function("relu1Quant8", &binding_utils::relu1Quant8Wrapper, allow_raw_pointers());
  function("relu6Quant8", &binding_utils::relu6Quant8Wrapper, allow_raw_pointers());
  function("logisticQuant8", &binding_utils::logisticQuant8Wrapper, allow_raw_pointers());
  function("fullyConnectedQuant8", &binding_utils::fullyConnectedQuant8Wrapper, allow_raw_pointers());
  function("concatenationQuant8", &binding_utils::concatenationQuant8Wrapper, allow_raw_pointers());
  function("l2normFloat32", &binding_utils::l2normFloat32Wrapper, allow_raw_pointers());
  function("l2normQuant8", &binding_utils::l2normQuant8Wrapper, allow_raw_pointers());
  function("localResponseNormFloat32", &binding_utils::localResponseNormFloat32Wrapper, allow_raw_pointers());
  function("depthToSpaceGeneric", &binding_utils::depthToSpaceGenericWrapper, allow_raw_pointers());
  function("spaceToDepthGeneric", &binding_utils::spaceToDepthGenericWrapper, allow_raw_pointers());
  */
}