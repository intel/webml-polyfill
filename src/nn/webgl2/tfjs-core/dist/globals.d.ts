import { Environment } from './environment';
export { customGrad, grad, grads, valueAndGrad, valueAndGrads, variableGrads } from './gradients';
export declare const tidy: typeof Environment.tidy;
export declare const keep: typeof Environment.keep;
export declare const dispose: typeof Environment.dispose;
export declare const time: typeof Environment.time;
export declare const profile: typeof Environment.profile;
