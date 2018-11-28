import { Features } from './environment_util';
import { KernelBackend } from './kernels/backend';
import { MathBackendCPU } from './kernels/backend_cpu';
export declare function envSatisfiesConstraints(constraints: Features): boolean;
export declare function parseKarmaFlags(args: string[]): TestEnv;
export declare function describeWithFlags(name: string, constraints: Features, tests: (env: TestEnv) => void): void;
export interface TestEnv {
    name: string;
    factory: () => KernelBackend;
    features: Features;
}
export declare let TEST_ENVS: TestEnv[];
export declare const CPU_FACTORY: () => MathBackendCPU;
export declare function setTestEnvs(testEnvs: TestEnv[]): void;
