#include <emscripten/bind.h>

#include "Operations.h"
#include "OperationsUtils.h"

using namespace emscripten;
using namespace nn;

namespace binding_utils {
  int32_t getShapeType(const Shape& shape) {
    return (int32_t)shape.type;
  }

  void setShapeType(Shape& shape, int32_t type) {
    shape.type = (OperandType)type;
  }

  val getShapeDimensions(const Shape& shape) {
    emscripten::val js_dims = emscripten::val::array();
    for (int i = 0; i < shape.dimensions.size(); i++) {
      js_dims.call<void>("push", shape.dimensions[i]);
    }
    return js_dims;
  }

  void setShapeDimensions(Shape& shape, val js_dims) {
    shape.dimensions = vecFromJSArray<uint32_t>(js_dims);
  }

  // Operation helper wrappers.
  bool addMulPrepareWrapper(const Shape& in1, const Shape& in2, Shape& out1) {
    return addMulPrepare(in1, in2, &out1);
  }

  bool floorPrepareWrapper(const Shape& input, Shape& output) {
    return floorPrepare(input, &output);
  }

  bool dequantizePrepareWrapper(const Shape& input, Shape& output) {
    return dequantizePrepare(input, &output);
  }

  bool depthwiseConvPrepareWrapper(const Shape& input, const Shape& filter, const Shape& bias,
                                   int32_t padding_left, int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                                   int32_t stride_width, int32_t stride_height, Shape& output) {
    return depthwiseConvPrepare(input, filter, bias, padding_left, padding_right, padding_top, padding_bottom, stride_width, stride_height, &output);
  }

  bool convPrepareWrapper(const Shape& input, const Shape& filter, const Shape& bias,
                          int32_t padding_left, int32_t padding_right, int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height, Shape& output) {
    return convPrepare(input, filter, bias, padding_left, padding_right, padding_top, padding_bottom, stride_width, stride_height, &output);
  }

  bool genericPoolingPrepareWrapper(const Shape& input,
                                    int32_t padding_left, int32_t padding_right,
                                    int32_t padding_top, int32_t padding_bottom,
                                    int32_t stride_width, int32_t stride_height,
                                    int32_t filter_width, int32_t filter_height,
                                    Shape& output) {
    return genericPoolingPrepare(input, padding_left, padding_right, padding_top, padding_bottom,
                                 stride_width, stride_height, filter_width, filter_height, &output);
  }

  bool genericActivationPrepareWrapper(const Shape& input, Shape& output) {
    return genericActivationPrepare(input, &output);
  }

  bool reshapePrepareWrapper(const Shape& input, const intptr_t targetDims, const int32_t targetDimsSize, Shape& output) {
    return reshapePrepare(input, (const int32_t*)targetDims, targetDimsSize, &output);
  }

  bool concatenationPrepareWrapper(const std::vector<Shape>& inputShapes,
                            int32_t axis,
                            Shape& output) {
    return concatenationPrepare(inputShapes, axis, &output);
  }
  bool fullyConnectedPrepareWrapper(const Shape& input,
                                    const Shape& weights,
                                    const Shape& bias,
                                    Shape& output) {
    return fullyConnectedPrepare(input, weights, bias, &output);
  }
  bool resizeBilinearPrepareWrapper(const Shape& input,
                                    int32_t height,
                                    int32_t width,
                                    Shape& output) {
    return resizeBilinearPrepare(input, height, width, &output);
  }

  // Operation wrappers.
  bool addFloat32Wrapper(const intptr_t in1, const Shape& shape1,
                         const intptr_t in2, const Shape& shape2,
                         int32_t activation, intptr_t out, const Shape& shapeOut) {
    return addFloat32((const float*)in1, shape1, (const float*)in2, shape2, activation, (float*)out, shapeOut);
  }

  bool mulFloat32Wrapper(const intptr_t in1, const Shape& shape1,
                         const intptr_t in2, const Shape& shape2,
                         int32_t activation, intptr_t out, const Shape& shapeOut) {
    return mulFloat32((const float*)in1, shape1, (const float*)in2, shape2, activation, (float*)out, shapeOut);
  }

  bool floorFloat32Wrapper(const intptr_t inputData, intptr_t outputData, const Shape& shape) {
    return floorFloat32((const float*)inputData, (float*)outputData, shape);
  }

  bool depthwiseConvFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                   const intptr_t filterData, const Shape& filterShape,
                                   const intptr_t biasData, const Shape& biasShape,
                                   int32_t padding_left, int32_t padding_right,
                                   int32_t padding_top, int32_t padding_bottom,
                                   int32_t stride_width, int32_t stride_height,
                                   int32_t depth_multiplier, int32_t activation,
                                   intptr_t outputData, const Shape& outputShape) {
    return depthwiseConvFloat32((const float*)inputData, inputShape,
                                (const float*)filterData, filterShape,
                                (const float*)biasData, biasShape,
                                padding_left, padding_right, padding_top, padding_bottom,
                                stride_width, stride_height,
                                depth_multiplier, activation,
                                (float*)outputData, outputShape);
  }

  bool convFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                          const intptr_t filterData, const Shape& filterShape,
                          const intptr_t biasData, const Shape& biasShape,
                          int32_t padding_left, int32_t padding_right,
                          int32_t padding_top, int32_t padding_bottom,
                          int32_t stride_width, int32_t stride_height,
                          int32_t activation,
                          intptr_t outputData, const Shape& outputShape) {
    return convFloat32((const float*)inputData, inputShape,
                       (const float*)filterData, filterShape,
                       (const float*)biasData, biasShape,
                       padding_left, padding_right, padding_top, padding_bottom,
                       stride_width, stride_height, activation,
                       (float*)outputData, outputShape);
  }

  bool averagePoolFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                 int32_t padding_left, int32_t padding_right,
                                 int32_t padding_top, int32_t padding_bottom,
                                 int32_t stride_width, int32_t stride_height,
                                 int32_t filter_width, int32_t filter_height, int32_t activation,
                                 intptr_t outputData, const Shape& outputShape) {
    return averagePoolFloat32((const float*)inputData, inputShape,
                              padding_left, padding_right,
                              padding_top, padding_bottom,
                              stride_width, stride_height,
                              filter_width, filter_height, activation,
                              (float*)outputData, outputShape);
  }

  bool maxPoolFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                 int32_t padding_left, int32_t padding_right,
                                 int32_t padding_top, int32_t padding_bottom,
                                 int32_t stride_width, int32_t stride_height,
                                 int32_t filter_width, int32_t filter_height, int32_t activation,
                                 intptr_t outputData, const Shape& outputShape) {
    return maxPoolFloat32((const float*)inputData, inputShape,
                          padding_left, padding_right,
                          padding_top, padding_bottom,
                          stride_width, stride_height,
                          filter_width, filter_height, activation,
                          (float*)outputData, outputShape);
  }

  bool softmaxFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                             const float beta,
                             intptr_t outputData, const Shape& outputShape) {
    return softmaxFloat32((const float*)inputData, inputShape,
                          beta,
                          (float*)outputData, outputShape);
  }

  bool reshapeGenericWrapper(const intptr_t inputData, const Shape& inputShape,
                             intptr_t outputData, const Shape& outputShape) {
    return reshapeGeneric((const void*)inputData, inputShape,
                          (void*)outputData, outputShape);
  }

  bool concatenationFloat32Wrapper(const std::vector<intptr_t>& inputDataPtrs,
                                   const std::vector<Shape>& inputShapes, int32_t axis,
                                   intptr_t outputData, const Shape& outputShape) {
    return concatenationFloat32((const std::vector<const float*>&)inputDataPtrs, inputShapes,
                                axis, (float*)outputData, outputShape);
  }

  bool fullyConnectedFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                    const intptr_t weightsData, const Shape& weightsShape,
                                    const intptr_t biasData, const Shape& biasShape,
                                    int32_t activation,
                                    intptr_t outputData, const Shape& outputShape) {
    return fullyConnectedFloat32((const float*) inputData, inputShape,
                                 (const float*) weightsData, weightsShape,
                                 (const float*) biasData, biasShape,
                                 activation,
                                 (float*) outputData, outputShape);
  }

  bool resizeBilinearFloat32Wrapper(const intptr_t inputData, const Shape& inputShape,
                                    intptr_t outputData, const Shape& outputShape) {
    return resizeBilinearFloat32((const float*) inputData, inputShape,
                                 (float*) outputData, outputShape);
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

  class_<Shape>("Shape")
    .constructor<>()
    .property("type", &binding_utils::getShapeType, &binding_utils::setShapeType)
    .property("dimensions", &binding_utils::getShapeDimensions, &binding_utils::setShapeDimensions)
    .property("scale", &Shape::scale)
    .property("offset", &Shape::offset)
    ;

  register_vector<Shape>("VectorShape");
  register_vector<intptr_t>("VectorPtr");

  // Operation helpers.
  function("addMulPrepare", &binding_utils::addMulPrepareWrapper);
  function("floorPrepare", &binding_utils::floorPrepareWrapper);
  function("dequantizePrepare", &binding_utils::dequantizePrepareWrapper);
  function("depthwiseConvPrepare", &binding_utils::depthwiseConvPrepareWrapper);
  function("convPrepare", &binding_utils::convPrepareWrapper);
  function("genericPoolingPrepare", &binding_utils::genericPoolingPrepareWrapper);
  function("genericActivationPrepare", &binding_utils::genericActivationPrepareWrapper);
  function("reshapePrepare", &binding_utils::reshapePrepareWrapper);
  function("concatenationPrepare", &binding_utils::concatenationPrepareWrapper);
  function("fullyConnectedPrepare", &binding_utils::fullyConnectedPrepareWrapper);
  function("resizeBilinearPrepare", &binding_utils::resizeBilinearPrepareWrapper);

  // Operations.
  function("addFloat32", &binding_utils::addFloat32Wrapper, allow_raw_pointers());
  function("mulFloat32", &binding_utils::mulFloat32Wrapper, allow_raw_pointers());
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