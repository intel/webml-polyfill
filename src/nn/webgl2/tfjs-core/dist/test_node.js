"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("./jasmine_util");
var backend_cpu_1 = require("./kernels/backend_cpu");
var jasmine = require('jasmine');
process.on('unhandledRejection', function (e) {
    throw e;
});
jasmine_util_1.setTestEnvs([{ name: 'node', factory: function () { return new backend_cpu_1.MathBackendCPU(); }, features: {} }]);
var runner = new jasmine();
runner.loadConfig({ spec_files: ['src/**/**_test.ts'], random: false });
runner.execute();
//# sourceMappingURL=test_node.js.map