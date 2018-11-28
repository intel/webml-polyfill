import { Tensor } from './tensor';
import { NamedTensorMap, TensorInfo } from './tensor_types';
export interface ModelPredictConfig {
    batchSize?: number;
    verbose?: boolean;
}
export interface InferenceModel {
    readonly inputs: TensorInfo[];
    readonly outputs: TensorInfo[];
    predict(inputs: Tensor | Tensor[] | NamedTensorMap, config: ModelPredictConfig): Tensor | Tensor[] | NamedTensorMap;
    execute(inputs: Tensor | Tensor[] | NamedTensorMap, outputs: string | string[]): Tensor | Tensor[];
}
