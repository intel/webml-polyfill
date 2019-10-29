import {OperationCode, OperandCode, PaddingCode, PreferenceCode, FuseCode} from './Enums'

export const operandCodeToTypedArrayMap = new Map([
  [OperandCode.FLOAT32, Float32Array],
  [OperandCode.INT32, Int32Array],
  [OperandCode.UINT32, Uint32Array],
  [OperandCode.TENSOR_FLOAT32, Float32Array],
  [OperandCode.TENSOR_INT32, Int32Array],
  [OperandCode.TENSOR_QUANT8_ASYMM, Uint8Array]
]);

export function isTensor(type) {
  if (type === OperandCode.TENSOR_FLOAT32 || type === OperandCode.TENSOR_INT32 || type === OperandCode.TENSOR_QUANT8_ASYMM) {
    return true;
  } else {
    return false;
  }
}

export function sizeOfTensorData(type, dims) {
  return operandCodeToTypedArrayMap.get(type).BYTES_PER_ELEMENT * product(dims);
}

export function sizeOfScalarData(type) {
  return operandCodeToTypedArrayMap.get(type).BYTES_PER_ELEMENT * 1;
}

export function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}

export function validateEnum(enumValue, enumType) {
  for (let k in enumType) {
    if (enumValue === enumType[k]) {
      return true;
    }
  }
  return false;
}

export function findKey(obj, value) {
  return Object.keys(obj).find(key => obj[key] === value);
}

export function stringifySubgraphCompact(model, ops) {
  return Object.entries(ops
      .map((opId) => findKey(OperationCode, model._operations[opId].type))
      .reduce((cnt, v) => {
        cnt[v] ? cnt[v]++ : cnt[v] = 1;
        return cnt;
      }, {}))
    .map(n => `${n[0]} x ${n[1]}`)
    .join(', ');
}
