import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'

export const operandCodeToTypedArrayMap = new Map([
  [OperandCode.tensor_float32, Float32Array],
  [OperandCode.tensor_int32, Int32Array],
  [OperandCode.tensor_quant8_asymm, Int8Array]
]);

export function isTensor(type) {
  let enumValue = OperandCode.enumValueOf(type);
  if (enumValue === OperandCode.tensor_float32 || enumValue === OperandCode.tensor_int32 || enumValue === OperandCode.tensor_quant8_asymm) {
    return true;
  } else {
    return false;
  }
}

export function sizeOfTensorData(type, dims) {
  let enumValue = OperandCode.enumValueOf(type);
  return operandCodeToTypedArrayMap.get(enumValue).BYTES_PER_ELEMENT * product(dims);
}

export function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}
