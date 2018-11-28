import { Tensor } from './tensor';
import { NamedTensorMap } from './tensor_types';
export interface TapeNode {
    id: number;
    name: string;
    outputs: Tensor[];
    inputs: NamedTensorMap;
    gradient?: (dy: Tensor | Tensor[]) => NamedGradientMap;
}
export declare type NamedGradientMap = {
    [inputName: string]: () => Tensor;
};
export declare function getFilteredNodesXToY(tape: TapeNode[], xs: Tensor[], y: Tensor): TapeNode[];
export declare function backpropagateGradients(tensorAccumulatedGradientMap: {
    [tensorId: number]: Tensor;
}, filteredTape: TapeNode[]): void;
