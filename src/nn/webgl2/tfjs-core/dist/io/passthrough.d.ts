import { IOHandler, ModelArtifacts, SaveResult, WeightsManifestEntry } from './types';
export declare function fromMemory(modelTopology: {}, weightSpecs?: WeightsManifestEntry[], weightData?: ArrayBuffer): IOHandler;
export declare function withSaveHandler(saveHandler: (artifacts: ModelArtifacts) => Promise<SaveResult>): IOHandler;
