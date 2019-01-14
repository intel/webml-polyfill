#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "external/tensorflow/tensorflow/contrib/lite/kernels/internal/types.h"
#include "external/tensorflow/tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "external/tensorflow/tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "external/tensorflow/tensorflow/contrib/lite/kernels/internal/optimized/depthwiseconv_float.h"

#include <vector>
#include <cmath>
#include <iostream>

using namespace emscripten;
using namespace tflite;

namespace binding_utils {
  // Operation wrappers.
  bool addFloat32Wrapper(const ArithmeticParams& op_params,
                         const RuntimeShape& input1_shape, 
                         const intptr_t input1_data, 
                         const RuntimeShape& input2_shape, 
                         const intptr_t input2_data, 
                         const RuntimeShape& output_shape, 
                         intptr_t output_data) {
    optimized_ops::Add(op_params,
                       input1_shape, (const float*) input1_data,
                       input2_shape, (const float*) input2_data,
                       output_shape, (float*) output_data);
  }

  bool broadCastAddFloat32Wrapper(const ArithmeticParams& op_params,
                                  const RuntimeShape& input1_shape, 
                                  const intptr_t input1_data, 
                                  const RuntimeShape& input2_shape, 
                                  const intptr_t input2_data, 
                                  const RuntimeShape& output_shape, 
                                  intptr_t output_data) {
    optimized_ops::BroadcastAdd4DSlow(op_params,
                                      input1_shape, (const float*) input1_data,
                                      input2_shape, (const float*) input2_data,
                                      output_shape, (float*) output_data);
  }

  bool mulFloat32Wrapper(const ArithmeticParams& op_params,
                         const RuntimeShape& input1_shape, 
                         const intptr_t input1_data, 
                         const RuntimeShape& input2_shape, 
                         const intptr_t input2_data, 
                         const RuntimeShape& output_shape, 
                         intptr_t output_data) {
    optimized_ops::Mul(op_params,
                       input1_shape, (const float*) input1_data,
                       input2_shape, (const float*) input2_data,
                       output_shape, (float*) output_data);
  }

  bool broadCastMulFloat32Wrapper(const ArithmeticParams& op_params,
                                  const RuntimeShape& input1_shape, 
                                  const intptr_t input1_data, 
                                  const RuntimeShape& input2_shape, 
                                  const intptr_t input2_data, 
                                  const RuntimeShape& output_shape, 
                                  intptr_t output_data) {
    optimized_ops::BroadcastMul4DSlow(op_params,
                                      input1_shape, (const float*) input1_data,
                                      input2_shape, (const float*) input2_data,
                                      output_shape, (float*) output_data);
  }

  bool floorFloat32Wrapper(const RuntimeShape& input_shape, 
                           const intptr_t inputData, 
                           const RuntimeShape& output_shape, 
                           intptr_t outputData) {
    optimized_ops::Floor(input_shape, (const float*)inputData,
                         output_shape, (float*)outputData);
  }

  bool depthwiseConvFloat32Wrapper(const DepthwiseParams& op_params,
                                   const RuntimeShape& inputShape, 
                                   const intptr_t inputData, 
                                   const RuntimeShape& filterShape, 
                                   const intptr_t filterData, 
                                   const RuntimeShape& biasShape, 
                                   const intptr_t biasData, 
                                   const RuntimeShape& outputShape, 
                                   intptr_t outputData) {
    optimized_ops::DepthwiseConv(op_params,
                                 inputShape, (const float*)inputData, 
                                 filterShape, (const float*)filterData, 
                                 biasShape, (const float*)biasData, 
                                 outputShape, (float*)outputData);
  }

  bool convFloat32Wrapper(const ConvParams& op_params, 
                          const RuntimeShape& inputShape, 
                          const intptr_t inputData, 
                          const RuntimeShape& filterShape, 
                          const intptr_t filterData, 
                          const RuntimeShape& biasShape, 
                          const intptr_t biasData, 
                          const RuntimeShape& outputShape, 
                          intptr_t outputData,
                          const RuntimeShape& im2colShape, 
                          intptr_t im2colData) {
    optimized_ops::Conv(op_params, 
                        inputShape, (const float*)inputData, 
                        filterShape, (const float*)filterData, 
                        biasShape, (const float*)biasData, 
                        outputShape, (float*)outputData, 
                        im2colShape, (float*)im2colData);
  }

  bool averagePoolFloat32Wrapper(const PoolParams op_params,
                                 const RuntimeShape& inputShape, 
                                 const intptr_t inputData, 
                                 const RuntimeShape& outputShape, 
                                 intptr_t outputData) {
    optimized_ops::AveragePool(op_params,
                               inputShape, (const float*)inputData,
                               outputShape, (float*)outputData);
  }

  bool maxPoolFloat32Wrapper(const PoolParams op_params,
                             const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    optimized_ops::MaxPool(op_params,
                           inputShape, (const float*)inputData,
                           outputShape, (float*)outputData);
  }

  bool softmaxFloat32Wrapper(const SoftmaxParams op_params,
                             const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    optimized_ops::Softmax(op_params, inputShape, (const float*)inputData,
                           outputShape, (float*)outputData);
  }

  bool reshapeFloat32Wrapper(const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    // implement it by self due to no reshape op in tflite::optimized_ops
    uint32_t size_count = (uint32_t)(inputShape.FlatSize() * sizeof(float));
    memcpy((float*)outputData, (const float*)inputData, size_count);
  }

  bool concatenationFloat32Wrapper(const ConcatenationParams op_params,  
                                   const std::vector<RuntimeShape*> inputShapes, 
                                   const std::vector<intptr_t>& inputDataPtrs,
                                   const RuntimeShape& outputShape, 
                                   intptr_t outputData) {
    optimized_ops::Concatenation<float>(op_params,
                                        inputShapes.data(),
                                        ((const std::vector<const float*>&)inputDataPtrs).data(), 
                                        outputShape, (float*)outputData);
  }

  bool fullyConnectedFloat32Wrapper(const FullyConnectedParams op_params,
                                    const RuntimeShape& inputShape, 
                                    const intptr_t inputData, 
                                    const RuntimeShape& weightsShape, 
                                    const intptr_t weightsData, 
                                    const RuntimeShape& biasShape, 
                                    const intptr_t biasData, 
                                    const RuntimeShape& outputShape, 
                                    intptr_t outputData) {
    optimized_ops::FullyConnected(op_params, 
                                  inputShape, (const float*)inputData, 
                                  weightsShape, (const float*)weightsData, 
                                  biasShape, (const float*)biasData,
                                  outputShape, (float*)outputData);
  }

  bool resizeBilinearFloat32Wrapper(const ResizeBilinearParams op_params,
                                    const RuntimeShape& inputShape, 
                                    const intptr_t inputData, 
                                    const RuntimeShape& outSizeShape, 
                                    const intptr_t outSizeData,
                                    const RuntimeShape& outputShape, 
                                    intptr_t outputData) {
    optimized_ops::ResizeBilinear(op_params, 
                                  inputShape, (const float*)inputData, 
                                  outSizeShape, (const int32_t*)outSizeData, 
                                  outputShape, (float*)outputData);
  }

}

EMSCRIPTEN_BINDINGS(nn)
{
  constant("MAX", std::numeric_limits<float>::max());
  constant("LOWEST", std::numeric_limits<float>::lowest());
  constant("MIN", std::numeric_limits<float>::min());

  class_<RuntimeShape>("RuntimeShape")
    .constructor<int>(select_overload<RuntimeShape(int)>([](int dimensions_count) {
        return RuntimeShape(dimensions_count);
      }
    ))
    .function("DimensionsCount", &RuntimeShape::DimensionsCount)
    .function("Dims", &RuntimeShape::Dims)
    .function("SetDim", &RuntimeShape::SetDim)
    ;

  value_object<PaddingValues>("PaddingValues")
    .field("width", &PaddingValues::width)
    .field("height", &PaddingValues::height)
    ;

  value_object<ConvParams>("ConvParams")
    .field("padding_values", &ConvParams::padding_values)
    .field("stride_width", &ConvParams::stride_width)
    .field("stride_height", &ConvParams::stride_height)
    .field("dilation_width_factor", &ConvParams::dilation_width_factor)
    .field("dilation_height_factor", &ConvParams::dilation_height_factor)
    .field("float_activation_min", &ConvParams::float_activation_min)
    .field("float_activation_max", &ConvParams::float_activation_max)
    ;

  value_object<DepthwiseParams>("DepthwiseParams")
    .field("padding_values", &DepthwiseParams::padding_values)
    .field("stride_width", &DepthwiseParams::stride_width)
    .field("stride_height", &DepthwiseParams::stride_height)
    .field("dilation_width_factor", &DepthwiseParams::dilation_width_factor)
    .field("dilation_height_factor", &DepthwiseParams::dilation_height_factor)
    .field("float_activation_min", &DepthwiseParams::float_activation_min)
    .field("float_activation_max", &DepthwiseParams::float_activation_max)
    .field("depth_multiplier", &DepthwiseParams::depth_multiplier)
    ;

  value_object<SoftmaxParams>("SoftmaxParams")
    .field("beta", &SoftmaxParams::beta)
    ;

  value_object<PoolParams>("PoolParams")
    .field("padding_values", &PoolParams::padding_values)
    .field("stride_width", &PoolParams::stride_width)
    .field("stride_height", &PoolParams::stride_height)
    .field("filter_width", &PoolParams::filter_width)
    .field("filter_height", &PoolParams::filter_height)
    .field("float_activation_min", &PoolParams::float_activation_min)
    .field("float_activation_max", &PoolParams::float_activation_max)
    ;

  value_object<ResizeBilinearParams>("ResizeBilinearParams")
    .field("align_corners", &ResizeBilinearParams::align_corners)
    ;

  value_object<ConcatenationParams>("ConcatenationParams")
    .field("axis", &ConcatenationParams::axis)
    .field("inputs_count", &ConcatenationParams::inputs_count)
    ;

  value_object<FullyConnectedParams>("FullyConnectedParams")
    .field("float_activation_min", &FullyConnectedParams::float_activation_min)
    .field("float_activation_max", &FullyConnectedParams::float_activation_max)
    ;

  value_object<ArithmeticParams>("ArithmeticParams")
    .field("float_activation_min", &ArithmeticParams::float_activation_min)
    .field("float_activation_max", &ArithmeticParams::float_activation_max)
    ;

  register_vector<RuntimeShape*>("VectorShape");
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
  function("reshapeFloat32", &binding_utils::reshapeFloat32Wrapper, allow_raw_pointers());
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