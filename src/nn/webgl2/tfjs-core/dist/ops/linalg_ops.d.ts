import { Tensor, Tensor1D, Tensor2D } from '../tensor';
declare function gramSchmidt_(xs: Tensor1D[] | Tensor2D): Tensor1D[] | Tensor2D;
declare function qr_(x: Tensor, fullMatrices?: boolean): [Tensor, Tensor];
export declare const gramSchmidt: typeof gramSchmidt_;
export declare const qr: typeof qr_;
export {};
