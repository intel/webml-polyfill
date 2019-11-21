#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "external/tensorflow/tensorflow/lite/kernels/cpu_backend_context.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/types.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/depthwiseconv_float.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/prelu.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "external/tensorflow/tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "fixedpoint/fixedpoint.h"
#include "public/gemmlowp.h"

#include <vector>
#include <cmath>
#include <iostream>

using namespace emscripten;
using namespace tflite;

namespace binding_utils {
  static gemmlowp::GemmContext gemm_context;
  static CpuBackendContext cpu_backend_context;
  static CpuFlags cpu_flags;
  
  // help functions
  void set_gemm_context_threads_num(int threads_num) {
    gemm_context.set_max_num_threads(threads_num);
  }

  void set_cpu_context_threads_num(int max_num_threads) {
    cpu_backend_context.SetMaxNumThreads(max_num_threads);
  }

  // Operation Implements.	
  template<typename T>
  void Maximum(const RuntimeShape& input1_shape, const T* input1_data,
               const T* input2_data, const RuntimeShape& output_shape,
               T* output_data) {
    auto input1_map = optimized_ops::MapAsVector(input1_data, input1_shape);
    auto input2_map = optimized_ops::MapAsVector(input2_data, output_shape);
    auto output_map = optimized_ops::MapAsVector(output_data, output_shape);
    output_map.array() = input1_map.array().max(input2_map.array());
  }

  template <typename T>
  T ApplyPrelu(T input, T alpha) {
    return input >= 0.0 ? input : input * alpha;
  }  

  // Operation wrappers.
  void addFloat32Wrapper(const ArithmeticParams& op_params,
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

  void addUint8Wrapper(const ArithmeticParams& op_params,
                       const RuntimeShape& input1_shape, 
                       const intptr_t input1_data, 
                       const RuntimeShape& input2_shape, 
                       const intptr_t input2_data, 
                       const RuntimeShape& output_shape, 
                       intptr_t output_data) {
    optimized_ops::Add(op_params,
                       input1_shape, (const uint8_t*) input1_data,
                       input2_shape, (const uint8_t*) input2_data,
                       output_shape, (uint8_t*) output_data);
  }

  void broadCastAddFloat32Wrapper(const ArithmeticParams& op_params,
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

  void mulFloat32Wrapper(const ArithmeticParams& op_params,
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

  void broadCastMulFloat32Wrapper(const ArithmeticParams& op_params,
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

  void floorFloat32Wrapper(const RuntimeShape& input_shape, 
                           const intptr_t inputData, 
                           const RuntimeShape& output_shape, 
                           intptr_t outputData) {
    optimized_ops::Floor(input_shape, (const float*)inputData,
                         output_shape, (float*)outputData);
  }

  void depthwiseConvFloat32Wrapper(const DepthwiseParams& op_params,
                                   const RuntimeShape& inputShape, 
                                   const intptr_t inputData, 
                                   const RuntimeShape& filterShape, 
                                   const intptr_t filterData, 
                                   const RuntimeShape& biasShape, 
                                   const intptr_t biasData, 
                                   const RuntimeShape& outputShape, 
                                   intptr_t outputData) {
    tflite::GetCpuFlags(&cpu_backend_context, &cpu_flags);
    optimized_ops::DepthwiseConv(op_params, inputShape,
                                 (const float*)inputData, filterShape,
                                 (const float*)filterData, biasShape,
                                 (const float*)biasData, outputShape,
                                 (float*)outputData, cpu_flags);
  }

  void depthwiseConvUint8Wrapper(const DepthwiseParams& op_params,
                                 const RuntimeShape& inputShape, 
                                 const intptr_t inputData, 
                                 const RuntimeShape& filterShape, 
                                 const intptr_t filterData, 
                                 const RuntimeShape& biasShape, 
                                 const intptr_t biasData, 
                                 const RuntimeShape& outputShape, 
                                 intptr_t outputData) {
    optimized_ops::DepthwiseConv(op_params, inputShape, 
                                 (const uint8_t*)inputData, filterShape,
                                 (const uint8_t*)filterData, biasShape, 
                                 (const int32_t*)biasData, outputShape,
                                 (uint8_t*)outputData, &gemm_context);
  }

  void convFloat32Wrapper(const ConvParams& op_params, 
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
    optimized_ops::Conv(op_params, inputShape,
                        (const float*)inputData, filterShape,
                        (const float*)filterData, biasShape,
                        (const float*)biasData, outputShape,
                        (float*)outputData, im2colShape,
                        (float*)im2colData, &cpu_backend_context);
  }

  void convUint8Wrapper(const ConvParams& op_params, 
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
    optimized_ops::Conv(op_params, inputShape,
                        (const uint8_t*)inputData, filterShape,
                        (const uint8_t*)filterData, biasShape,
                        (const int32_t*)biasData, outputShape,
                        (uint8_t*)outputData, im2colShape, 
                        (uint8_t*)im2colData, &cpu_backend_context);
  }

  void averagePoolFloat32Wrapper(const PoolParams op_params,
                                 const RuntimeShape& inputShape, 
                                 const intptr_t inputData, 
                                 const RuntimeShape& outputShape, 
                                 intptr_t outputData) {
    optimized_ops::AveragePool(op_params,
                               inputShape, (const float*)inputData,
                               outputShape, (float*)outputData);
  }
  
  void averagePoolUint8Wrapper(const PoolParams op_params,
                               const RuntimeShape& inputShape, 
                               const intptr_t inputData, 
                               const RuntimeShape& outputShape, 
                               intptr_t outputData) {
    optimized_ops::AveragePool(op_params,
                               inputShape, (const uint8_t*)inputData,
                               outputShape, (uint8_t*)outputData);
  }

  void maxPoolFloat32Wrapper(const PoolParams op_params,
                             const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    optimized_ops::MaxPool(op_params,
                           inputShape, (const float*)inputData,
                           outputShape, (float*)outputData);
  }

  void maxPoolUint8Wrapper(const PoolParams op_params,
                           const RuntimeShape& inputShape, 
                           const intptr_t inputData, 
                           const RuntimeShape& outputShape, 
                           intptr_t outputData) {
    optimized_ops::MaxPool(op_params,
                           inputShape, (const uint8_t*)inputData,
                           outputShape, (uint8_t*)outputData);
  }

  void softmaxFloat32Wrapper(const SoftmaxParams op_params,
                             const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    optimized_ops::Softmax(op_params, inputShape, (const float*)inputData,
                           outputShape, (float*)outputData);
  }

  void softmaxUint8Wrapper(const SoftmaxParams op_params,
                           const RuntimeShape& inputShape, 
                           const intptr_t inputData, 
                           const RuntimeShape& outputShape, 
                           intptr_t outputData) {
    optimized_ops::Softmax(op_params, inputShape, (const uint8_t*)inputData,
                           outputShape, (uint8_t*)outputData);
  }

  void reshapeFloat32Wrapper(const RuntimeShape& inputShape, 
                             const intptr_t inputData, 
                             const RuntimeShape& outputShape, 
                             intptr_t outputData) {
    // implement it by self due to no reshape op in tflite::optimized_ops
    uint32_t size_count = (uint32_t)(inputShape.FlatSize() * sizeof(float));
    memcpy((float*)outputData, (const float*)inputData, size_count);
  }

  void reshapeUint8Wrapper(const RuntimeShape& inputShape, 
                           const intptr_t inputData, 
                           const RuntimeShape& outputShape, 
                           intptr_t outputData) {
    // implement it by self due to no reshape op in tflite::optimized_ops
    uint32_t size_count = (uint32_t)(inputShape.FlatSize() * sizeof(uint8_t));
    memcpy((uint8_t*)outputData, (const uint8_t*)inputData, size_count);
  }

  void concatenationFloat32Wrapper(const ConcatenationParams& op_params, 
                                   const std::vector<RuntimeShape*> inputShapes, 
                                   const std::vector<intptr_t>& inputDataPtrs,
                                   const RuntimeShape& outputShape, 
                                   intptr_t outputData) {
    optimized_ops::Concatenation<float>(op_params,
                                        inputShapes.data(),
                                        ((const std::vector<const float*>&)inputDataPtrs).data(), 
                                        outputShape, (float*)outputData);
  }

  void concatenationUint8Wrapper(ConcatenationParams& op_params, 
                                 const std::vector<RuntimeShape*> inputShapes, 
                                 const std::vector<intptr_t>& inputDataPtrs,
                                 val inputScales,
                                 val inputZeroPoints,
                                 const RuntimeShape& outputShape, 
                                 intptr_t outputData) {
    const std::vector<float>& inputScaleRef = vecFromJSArray<float>(inputScales);
    const std::vector<int32_t>& inputZeroPointRef = vecFromJSArray<int32_t>(inputZeroPoints);
    op_params.input_scale = inputScaleRef.data();
    op_params.input_zeropoint = inputZeroPointRef.data();
    optimized_ops::ConcatenationWithScaling(op_params,
                                            inputShapes.data(),
                                            ((const std::vector<const uint8_t*>&)inputDataPtrs).data(), 
                                            outputShape, (uint8_t*)outputData);
  }

  void fullyConnectedFloat32Wrapper(const FullyConnectedParams op_params,
                                    const RuntimeShape& inputShape, 
                                    const intptr_t inputData, 
                                    const RuntimeShape& weightsShape, 
                                    const intptr_t weightsData, 
                                    const RuntimeShape& biasShape, 
                                    const intptr_t biasData, 
                                    const RuntimeShape& outputShape, 
                                    intptr_t outputData) {
    optimized_ops::FullyConnected(op_params, inputShape,
                                  (const float*)inputData, weightsShape,
                                  (const float*)weightsData, biasShape,
                                  (const float*)biasData, outputShape,
                                  (float*)outputData, &cpu_backend_context);
  }

  void fullyConnectedUint8Wrapper(const FullyConnectedParams op_params,
                                  const RuntimeShape& inputShape, 
                                  const intptr_t inputData, 
                                  const RuntimeShape& weightsShape, 
                                  const intptr_t weightsData, 
                                  const RuntimeShape& biasShape, 
                                  const intptr_t biasData, 
                                  const RuntimeShape& outputShape, 
                                  intptr_t outputData) {
    optimized_ops::FullyConnected(op_params, inputShape,
                                  (const uint8_t*)inputData, weightsShape,
                                  (const uint8_t*)weightsData, biasShape,
                                  (const int32_t*)biasData, outputShape,
                                  (uint8_t*)outputData, &cpu_backend_context);
  }

  void resizeBilinearFloat32Wrapper(const ResizeBilinearParams op_params,
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

  void tanhFloat32Wrapper(const RuntimeShape& inputShape, 
                          const intptr_t inputData, 
                          const RuntimeShape& outputShape, 
                          intptr_t outputData) {
    optimized_ops::Tanh(inputShape, (const float*)inputData, 
                        outputShape, (float*)outputData);
  }

  void maximumFloat32Wrapper(const RuntimeShape& input1_shape, 
                             const intptr_t input1_data,
                             const RuntimeShape& input2_shape,
                             const intptr_t input2_data, 
                             const RuntimeShape& output_shape,
                             intptr_t output_data) {
    binding_utils::Maximum(input1_shape, (const float*)input1_data,
                           (const float*)input2_data,
                           output_shape, (float*) output_data);
  }

  void batchToSpaceNDFloat32Wrapper(const RuntimeShape& unextended_input1_shape, 
                                    const intptr_t input1_data,
                                    const RuntimeShape& unextended_input2_shape, 
                                    const intptr_t block_shape_data,
                                    const RuntimeShape& unextended_input3_shape, 
                                    const intptr_t crops_data,
                                    const RuntimeShape& unextended_output_shape, 
                                    intptr_t output_data) {
    optimized_ops::BatchToSpaceND(unextended_input1_shape, (const float*) input1_data,
                                  unextended_input2_shape, (const int32_t*) block_shape_data,
                                  unextended_input3_shape, (const int32_t*) crops_data,
                                  unextended_output_shape, (float*) output_data);
  }

  void transposeFloat32Wrapper(const TransposeParams& op_params,
                               const RuntimeShape& unextended_input_shape, 
                               const intptr_t input_data,
                               const RuntimeShape& unextended_output_shape, 
                               intptr_t output_data) {
    optimized_ops::Transpose(op_params,
                             unextended_input_shape, (const float*) input_data,
                             unextended_output_shape, (float*) output_data);
  }

  void argMaxFloat32Wrapper(const RuntimeShape& input1_shape,
                            const intptr_t input1_data,
                            const intptr_t input2_data,
                            const RuntimeShape& output_shape,
                            intptr_t output_data) {
    optimized_ops::ArgMax(input1_shape, (const float*) input1_data,
                          (const int32_t*) input2_data, output_shape,
                          (int32_t*) output_data);
  }

  void logisticFloat32Wrapper(const RuntimeShape& input_shape,
                              const intptr_t inputData,
                              const RuntimeShape& output_shape,
                              intptr_t outputData) {
    optimized_ops::Logistic(input_shape, (const float*) inputData,
                            output_shape, (float*) outputData);                            
  }

  void logisticUint8Wrapper(const LogisticParams& params,
                            const RuntimeShape& input_shape,
                            const intptr_t input_data,
                            const RuntimeShape& output_shape,
                            intptr_t output_data) {
    optimized_ops::Logistic(params, input_shape, (const uint8_t*) input_data,
                            output_shape, (uint8_t*) output_data);
  }

  void preluFloat32Wrapper(const RuntimeShape& input_shape,
                           const intptr_t input_data,
                           const RuntimeShape& alpha_shape,
                           const intptr_t alpha_data,
                           const RuntimeShape& output_shape,
                           intptr_t output_data) {
    reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
                           input_shape, (const float*) input_data, 
                           alpha_shape, (const float*) alpha_data, 
                           output_shape, (float*) output_data, ApplyPrelu<float>);
  }

  void preluUint8Wrapper(const PreluParams& params,
                         const RuntimeShape& input_shape,
                         const intptr_t input_data,
                         const RuntimeShape& alpha_shape,
                         const intptr_t alpha_data,
                         const RuntimeShape& output_shape,
                         intptr_t output_data) {
    reference_ops::BroadcastPrelu4DSlow(params, 
                         input_shape, (const uint8_t*) input_data, 
                         alpha_shape, (const uint8_t*) alpha_data, 
                         output_shape, (uint8_t*) output_data);
  }
}

EMSCRIPTEN_BINDINGS(nn)
{
  constant("FLOAT_MAX", std::numeric_limits<float>::max());
  constant("FLOAT_LOWEST", std::numeric_limits<float>::lowest());
  constant("FLOAT_MIN", std::numeric_limits<float>::min());
  constant("UINT8_MAX", std::numeric_limits<uint8_t>::max());
  constant("UINT8_LOWEST", std::numeric_limits<uint8_t>::lowest());
  constant("UINT8_MIN", std::numeric_limits<uint8_t>::min());
  constant("INT32_MAX", std::numeric_limits<int32_t>::max());

  class_<RuntimeShape>("RuntimeShape")
    .constructor<int>()
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
    // float activation params.
    .field("float_activation_min", &ConvParams::float_activation_min)
    .field("float_activation_max", &ConvParams::float_activation_max)
    // uint8 inference params.
    .field("input_offset", &ConvParams::input_offset)
    .field("weights_offset", &ConvParams::weights_offset)
    .field("output_offset", &ConvParams::output_offset)
    .field("output_multiplier", &ConvParams::output_multiplier)
    .field("output_shift", &ConvParams::output_shift)
    // uint8, etc, activation params.
    .field("quantized_activation_min", &ConvParams::quantized_activation_min)
    .field("quantized_activation_max", &ConvParams::quantized_activation_max)
    ;

  value_object<DepthwiseParams>("DepthwiseParams")
    .field("padding_values", &DepthwiseParams::padding_values)
    .field("stride_width", &DepthwiseParams::stride_width)
    .field("stride_height", &DepthwiseParams::stride_height)
    .field("dilation_width_factor", &DepthwiseParams::dilation_width_factor)
    .field("dilation_height_factor", &DepthwiseParams::dilation_height_factor)
    .field("depth_multiplier", &DepthwiseParams::depth_multiplier)
    // float activation params.
    .field("float_activation_min", &DepthwiseParams::float_activation_min)
    .field("float_activation_max", &DepthwiseParams::float_activation_max)
    // uint8 inference params.
    .field("input_offset", &DepthwiseParams::input_offset)
    .field("weights_offset", &DepthwiseParams::weights_offset)
    .field("output_offset", &DepthwiseParams::output_offset)
    .field("output_multiplier", &DepthwiseParams::output_multiplier)
    .field("output_shift", &DepthwiseParams::output_shift)
    // uint8, etc, activation params.
    .field("quantized_activation_min", &DepthwiseParams::quantized_activation_min)
    .field("quantized_activation_max", &DepthwiseParams::quantized_activation_max)
    ;

  value_object<SoftmaxParams>("SoftmaxParams")
    .field("beta", &SoftmaxParams::beta)
    // uint8 inference params.  Used even when beta defaults to 1.0.
    .field("input_multiplier", &SoftmaxParams::input_multiplier)
    .field("input_left_shift", &SoftmaxParams::input_left_shift)
    .field("diff_min", &SoftmaxParams::diff_min)
    ;

  value_object<PoolParams>("PoolParams")
    .field("padding_values", &PoolParams::padding_values)
    .field("stride_width", &PoolParams::stride_width)
    .field("stride_height", &PoolParams::stride_height)
    .field("filter_width", &PoolParams::filter_width)
    .field("filter_height", &PoolParams::filter_height)
    // float activation params.
    .field("float_activation_min", &PoolParams::float_activation_min)
    .field("float_activation_max", &PoolParams::float_activation_max)
    // uint8, etc, activation params.
    .field("quantized_activation_min", &PoolParams::quantized_activation_min)
    .field("quantized_activation_max", &PoolParams::quantized_activation_max)
    ;

  value_object<ResizeBilinearParams>("ResizeBilinearParams")
    .field("align_corners", &ResizeBilinearParams::align_corners)
    ;

  value_object<ConcatenationParams>("ConcatenationParams")
    .field("axis", &ConcatenationParams::axis)
    .field("inputs_count", &ConcatenationParams::inputs_count)
    .field("output_scale", &ConcatenationParams::output_scale)
    .field("output_zeropoint", &ConcatenationParams::output_zeropoint)
    ;

  value_object<FullyConnectedParams>("FullyConnectedParams")
    // float activation params.
    .field("float_activation_min", &FullyConnectedParams::float_activation_min)
    .field("float_activation_max", &FullyConnectedParams::float_activation_max)
    // uint8 inference params.
    .field("input_offset", &FullyConnectedParams::input_offset)
    .field("weights_offset", &FullyConnectedParams::weights_offset)
    .field("output_offset", &FullyConnectedParams::output_offset)
    .field("output_multiplier", &FullyConnectedParams::output_multiplier)
    .field("output_shift", &FullyConnectedParams::output_shift)
    // uint8, etc, activation params.
    .field("quantized_activation_min", &FullyConnectedParams::quantized_activation_min)
    .field("quantized_activation_max", &FullyConnectedParams::quantized_activation_max)
    ;

  value_object<ArithmeticParams>("ArithmeticParams")
    // float activation params.
    .field("float_activation_min", &ArithmeticParams::float_activation_min)
    .field("float_activation_max", &ArithmeticParams::float_activation_max)
    // uint8 inference params.
    .field("input1_offset", &ArithmeticParams::input1_offset)
    .field("input2_offset", &ArithmeticParams::input2_offset)
    .field("output_offset", &ArithmeticParams::output_offset)
    .field("output_multiplier", &ArithmeticParams::output_multiplier)
    .field("output_shift", &ArithmeticParams::output_shift)
    // Add / Sub, not Mul, uint8 inference params.
    .field("left_shift", &ArithmeticParams::left_shift)
    .field("input1_multiplier", &ArithmeticParams::input1_multiplier)
    .field("input1_shift", &ArithmeticParams::input1_shift)
    .field("input2_multiplier", &ArithmeticParams::input2_multiplier)
    .field("input2_shift", &ArithmeticParams::input2_shift)
    // uint8, etc, activation params.
    .field("quantized_activation_min", &ArithmeticParams::quantized_activation_min)
    .field("quantized_activation_max", &ArithmeticParams::quantized_activation_max)
    ;

  value_object<TransposeParams>("TransposeParams")
    .field("perm", &TransposeParams::perm)
    .field("perm_count", &TransposeParams::perm_count)
    ;

  value_object<LogisticParams>("LogisticParams")
    // uint8 inference params.
    .field("input_zero_point", &LogisticParams::input_zero_point)
    .field("input_range_radius", &LogisticParams::input_range_radius)
    .field("input_multiplier", &LogisticParams::input_multiplier)
    .field("input_left_shift", &LogisticParams::input_left_shift)
    ;

  value_object<PreluParams>("PreluParams")
    .field("input_offset", &PreluParams::input_offset)
    .field("alpha_offset", &PreluParams::alpha_offset)
    .field("output_offset", &PreluParams::output_offset)
    .field("output_multiplier", &PreluParams::output_multiplier)
    .field("output_shift", &PreluParams::output_shift)
    ;
  
  value_array<std::array<int32_t, 4>>("array_int32_4")
    .element(emscripten::index<0>())
    .element(emscripten::index<1>())
    .element(emscripten::index<2>())
    .element(emscripten::index<3>())
    ;

  register_vector<RuntimeShape*>("VectorShape");
  register_vector<intptr_t>("VectorPtr");

  // help functions
  function("set_gemm_context_threads_num", &binding_utils::set_gemm_context_threads_num);
  function("set_cpu_context_threads_num", &binding_utils::set_cpu_context_threads_num);
  

  // Operations.
  function("addFloat32", &binding_utils::addFloat32Wrapper, allow_raw_pointers());
  function("addUint8", &binding_utils::addUint8Wrapper, allow_raw_pointers());
  function("broadCastAddFloat32", &binding_utils::broadCastAddFloat32Wrapper, allow_raw_pointers());
  function("mulFloat32", &binding_utils::mulFloat32Wrapper, allow_raw_pointers());
  function("broadCastMulFloat32", &binding_utils::broadCastMulFloat32Wrapper, allow_raw_pointers());
  function("floorFloat32", &binding_utils::floorFloat32Wrapper, allow_raw_pointers());
  function("depthwiseConvFloat32", &binding_utils::depthwiseConvFloat32Wrapper, allow_raw_pointers());
  function("depthwiseConvUint8", &binding_utils::depthwiseConvUint8Wrapper, allow_raw_pointers());
  function("convFloat32", &binding_utils::convFloat32Wrapper, allow_raw_pointers());
  function("convUint8", &binding_utils::convUint8Wrapper, allow_raw_pointers());
  function("averagePoolFloat32", &binding_utils::averagePoolFloat32Wrapper, allow_raw_pointers());
  function("averagePoolUint8", &binding_utils::averagePoolUint8Wrapper, allow_raw_pointers());
  function("softmaxFloat32", &binding_utils::softmaxFloat32Wrapper, allow_raw_pointers());
  function("softmaxUint8", &binding_utils::softmaxUint8Wrapper, allow_raw_pointers());
  function("reshapeFloat32", &binding_utils::reshapeFloat32Wrapper, allow_raw_pointers());
  function("reshapeUint8", &binding_utils::reshapeUint8Wrapper, allow_raw_pointers());
  function("maxPoolFloat32", &binding_utils::maxPoolFloat32Wrapper, allow_raw_pointers());
  function("maxPoolUint8", &binding_utils::maxPoolUint8Wrapper, allow_raw_pointers());
  function("concatenationFloat32", &binding_utils::concatenationFloat32Wrapper, allow_raw_pointers());
  function("concatenationUint8", &binding_utils::concatenationUint8Wrapper, allow_raw_pointers());
  function("fullyConnectedFloat32", &binding_utils::fullyConnectedFloat32Wrapper, allow_raw_pointers());
  function("fullyConnectedUint8", &binding_utils::fullyConnectedUint8Wrapper, allow_raw_pointers());
  function("resizeBilinearFloat32", &binding_utils::resizeBilinearFloat32Wrapper, allow_raw_pointers());
  function("tanhFloat32", &binding_utils::tanhFloat32Wrapper, allow_raw_pointers());
  function("maximumFloat32", &binding_utils::maximumFloat32Wrapper, allow_raw_pointers());
  function("batchToSpaceNDFloat32", &binding_utils::batchToSpaceNDFloat32Wrapper, allow_raw_pointers());
  function("transposeFloat32", &binding_utils::transposeFloat32Wrapper, allow_raw_pointers());
  function("argMaxFloat32", &binding_utils::argMaxFloat32Wrapper, allow_raw_pointers());
  function("logisticFloat32", &binding_utils::logisticFloat32Wrapper, allow_raw_pointers());
  function("logisticUint8", &binding_utils::logisticUint8Wrapper, allow_raw_pointers());
  function("preluFloat32", &binding_utils::preluFloat32Wrapper, allow_raw_pointers());
  function("preluUint8", &binding_utils::preluUint8Wrapper, allow_raw_pointers());

  // TODO: operation wrappers
  /*
  function("l2PoolFloat32", &binding_utils::l2PoolFloat32Wrapper, allow_raw_pointers());
  function("maxPoolQuant8", &binding_utils::maxPoolQuant8Wrapper, allow_raw_pointers());
  function("reluFloat32", &binding_utils::reluFloat32Wrapper, allow_raw_pointers());
  function("relu1Float32", &binding_utils::relu1Float32Wrapper, allow_raw_pointers());
  function("relu6Float32", &binding_utils::relu6Float32Wrapper, allow_raw_pointers());
  function("reluQuant8", &binding_utils::reluQuant8Wrapper, allow_raw_pointers());
  function("relu1Quant8", &binding_utils::relu1Quant8Wrapper, allow_raw_pointers());
  function("relu6Quant8", &binding_utils::relu6Quant8Wrapper, allow_raw_pointers());
  function("fullyConnectedQuant8", &binding_utils::fullyConnectedQuant8Wrapper, allow_raw_pointers());
  function("concatenationQuant8", &binding_utils::concatenationQuant8Wrapper, allow_raw_pointers());
  function("l2normFloat32", &binding_utils::l2normFloat32Wrapper, allow_raw_pointers());
  function("l2normQuant8", &binding_utils::l2normQuant8Wrapper, allow_raw_pointers());
  function("localResponseNormFloat32", &binding_utils::localResponseNormFloat32Wrapper, allow_raw_pointers());
  function("depthToSpaceGeneric", &binding_utils::depthToSpaceGenericWrapper, allow_raw_pointers());
  function("spaceToDepthGeneric", &binding_utils::spaceToDepthGenericWrapper, allow_raw_pointers());
  */
}
