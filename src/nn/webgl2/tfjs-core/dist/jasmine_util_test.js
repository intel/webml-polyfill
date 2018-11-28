"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("./index");
var jasmine_util_1 = require("./jasmine_util");
var backend_cpu_1 = require("./kernels/backend_cpu");
var backend_webgl_1 = require("./kernels/backend_webgl");
var test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('jasmine_util.envSatisfiesConstraints', {}, function () {
    it('ENV satisfies empty constraints', function () {
        expect(jasmine_util_1.envSatisfiesConstraints({})).toBe(true);
    });
    it('ENV satisfies matching constraints', function () {
        var c = { TEST_EPSILON: tf.ENV.get('TEST_EPSILON') };
        expect(jasmine_util_1.envSatisfiesConstraints(c)).toBe(true);
    });
    it('ENV does not satisfy mismatching constraints', function () {
        var c = { TEST_EPSILON: tf.ENV.get('TEST_EPSILON') + 0.1 };
        expect(jasmine_util_1.envSatisfiesConstraints(c)).toBe(false);
    });
});
describe('jasmine_util.parseKarmaFlags', function () {
    it('parse empty args', function () {
        var res = jasmine_util_1.parseKarmaFlags([]);
        expect(res).toBeNull();
    });
    it('--backend cpu', function () {
        var res = jasmine_util_1.parseKarmaFlags(['--backend', 'cpu']);
        expect(res.name).toBe('cpu');
        expect(res.features).toEqual({ 'HAS_WEBGL': false });
        expect(res.factory() instanceof backend_cpu_1.MathBackendCPU).toBe(true);
    });
    it('--backend cpu --features {"IS_NODE": true}', function () {
        var res = jasmine_util_1.parseKarmaFlags(['--backend', 'cpu', '--features', '{"IS_NODE": true}']);
        expect(res.name).toBe('cpu');
        expect(res.features).toEqual({ IS_NODE: true });
        expect(res.factory() instanceof backend_cpu_1.MathBackendCPU).toBe(true);
    });
    it('"--backend unknown" throws error', function () {
        expect(function () { return jasmine_util_1.parseKarmaFlags(['--backend', 'unknown']); }).toThrowError();
    });
    it('"--features {}" throws error since --backend is missing', function () {
        expect(function () { return jasmine_util_1.parseKarmaFlags(['--features', '{}']); }).toThrowError();
    });
    it('"--backend cpu --features" throws error since features value is missing', function () {
        expect(function () { return jasmine_util_1.parseKarmaFlags(['--backend', 'cpu', '--features']); })
            .toThrowError();
    });
    it('"--backend cpu --features notJson" throws error', function () {
        expect(function () { return jasmine_util_1.parseKarmaFlags(['--backend', 'cpu', '--features', 'notJson']); })
            .toThrowError();
    });
});
jasmine_util_1.describeWithFlags('jasmine_util.envSatisfiesConstraints', test_util_1.WEBGL_ENVS, function () {
    it('--backend webgl', function () {
        var res = jasmine_util_1.parseKarmaFlags(['--backend', 'webgl']);
        expect(res.name).toBe('webgl');
        expect(res.features).toEqual({});
        expect(res.factory() instanceof backend_webgl_1.MathBackendWebGL).toBe(true);
    });
});
//# sourceMappingURL=jasmine_util_test.js.map