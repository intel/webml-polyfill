import { CustomGradientFunc, ScopeFn } from './engine';
import { Scalar, Tensor, Variable } from './tensor';
import { NamedTensorMap, TensorContainer } from './tensor_types';
declare function gradScope<T extends TensorContainer>(nameOrScopeFn: string | ScopeFn<T>, scopeFn?: ScopeFn<T>): T;
declare function grad<I extends Tensor, O extends Tensor>(f: (x: I) => O): (x: I, dy?: O) => I;
declare function grads<O extends Tensor>(f: (...args: Tensor[]) => O): (args: Tensor[], dy?: O) => Tensor[];
declare function valueAndGrad<I extends Tensor, O extends Tensor>(f: (x: I) => O): (x: I, dy?: O) => {
    value: O;
    grad: I;
};
declare function valueAndGrads<O extends Tensor>(f: (...args: Tensor[]) => O): (args: Tensor[], dy?: O) => {
    grads: Tensor[];
    value: O;
};
declare function variableGrads(f: () => Scalar, varList?: Variable[]): {
    value: Scalar;
    grads: NamedTensorMap;
};
declare function customGrad<T extends Tensor>(f: CustomGradientFunc<T>): (...args: Tensor[]) => T;
export { gradScope, customGrad, variableGrads, valueAndGrad, valueAndGrads, grad, grads, };
