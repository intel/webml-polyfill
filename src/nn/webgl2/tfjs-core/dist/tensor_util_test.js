"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("./index");
var jasmine_util_1 = require("./jasmine_util");
var tensor_util_1 = require("./tensor_util");
var tensor_util_env_1 = require("./tensor_util_env");
var test_util_1 = require("./test_util");
describe('tensor_util.isTensorInList', function () {
    it('not in list', function () {
        var a = tf.scalar(1);
        var list = [tf.scalar(1), tf.tensor1d([1, 2, 3])];
        expect(tensor_util_1.isTensorInList(a, list)).toBe(false);
    });
    it('in list', function () {
        var a = tf.scalar(1);
        var list = [tf.scalar(2), tf.tensor1d([1, 2, 3]), a];
        expect(tensor_util_1.isTensorInList(a, list)).toBe(true);
    });
});
describe('tensor_util.flattenNameArrayMap', function () {
    it('basic', function () {
        var a = tf.scalar(1);
        var b = tf.scalar(3);
        var c = tf.tensor1d([1, 2, 3]);
        var map = { a: a, b: b, c: c };
        expect(tensor_util_1.flattenNameArrayMap(map, Object.keys(map))).toEqual([a, b, c]);
    });
});
describe('tensor_util.unflattenToNameArrayMap', function () {
    it('basic', function () {
        var a = tf.scalar(1);
        var b = tf.scalar(3);
        var c = tf.tensor1d([1, 2, 3]);
        expect(tensor_util_1.unflattenToNameArrayMap(['a', 'b', 'c'], [
            a, b, c
        ])).toEqual({ a: a, b: b, c: c });
    });
});
describe('getTensorsInContainer', function () {
    it('null input returns empty tensor', function () {
        var results = tensor_util_1.getTensorsInContainer(null);
        expect(results).toEqual([]);
    });
    it('tensor input returns one element tensor', function () {
        var x = tf.scalar(1);
        var results = tensor_util_1.getTensorsInContainer(x);
        expect(results).toEqual([x]);
    });
    it('name tensor map returns flattened tensor', function () {
        var x1 = tf.scalar(1);
        var x2 = tf.scalar(3);
        var x3 = tf.scalar(4);
        var results = tensor_util_1.getTensorsInContainer({ x1: x1, x2: x2, x3: x3 });
        expect(results).toEqual([x1, x2, x3]);
    });
    it('can extract from arbitrary depth', function () {
        var container = [
            { x: tf.scalar(1), y: tf.scalar(2) },
            [[[tf.scalar(3)]], { z: tf.scalar(4) }]
        ];
        var results = tensor_util_1.getTensorsInContainer(container);
        expect(results.length).toBe(4);
    });
    it('works with loops in container', function () {
        var container = [tf.scalar(1), tf.scalar(2), [tf.scalar(3)]];
        var innerContainer = [container];
        container.push(innerContainer);
        var results = tensor_util_1.getTensorsInContainer(container);
        expect(results.length).toBe(3);
    });
});
jasmine_util_1.describeWithFlags('convertToTensor', test_util_1.ALL_ENVS, function () {
    it('primitive integer, NaN converts to zero, no error thrown', function () {
        var a = function () { return tensor_util_env_1.convertToTensor(NaN, 'a', 'test', 'int32'); };
        expect(a).not.toThrowError();
        var b = tensor_util_env_1.convertToTensor(NaN, 'b', 'test', 'int32');
        expect(b.rank).toBe(0);
        expect(b.dtype).toBe('int32');
        test_util_1.expectNumbersClose(b.get(), 0);
    });
    it('primitive number', function () {
        var a = tensor_util_env_1.convertToTensor(3, 'a', 'test');
        expect(a.rank).toBe(0);
        expect(a.dtype).toBe('float32');
        test_util_1.expectNumbersClose(a.get(), 3);
    });
    it('primitive integer, NaN converts to zero', function () {
        var a = tensor_util_env_1.convertToTensor(NaN, 'a', 'test', 'int32');
        expect(a.rank).toBe(0);
        expect(a.dtype).toBe('int32');
        test_util_1.expectNumbersClose(a.get(), 0);
    });
    it('primitive boolean, parsed as float', function () {
        var a = tensor_util_env_1.convertToTensor(true, 'a', 'test');
        expect(a.rank).toBe(0);
        expect(a.dtype).toBe('float32');
        test_util_1.expectNumbersClose(a.get(), 1);
    });
    it('primitive boolean, parsed as bool', function () {
        var a = tensor_util_env_1.convertToTensor(true, 'a', 'test', 'bool');
        expect(a.rank).toBe(0);
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('array1d', function () {
        var a = tensor_util_env_1.convertToTensor([1, 2, 3], 'a', 'test');
        expect(a.rank).toBe(1);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        test_util_1.expectArraysClose(a, [1, 2, 3]);
    });
    it('array2d', function () {
        var a = tensor_util_env_1.convertToTensor([[1], [2], [3]], 'a', 'test');
        expect(a.rank).toBe(2);
        expect(a.shape).toEqual([3, 1]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [1, 2, 3]);
    });
    it('array3d', function () {
        var a = tensor_util_env_1.convertToTensor([[[1], [2]], [[3], [4]]], 'a', 'test');
        expect(a.rank).toBe(3);
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('array4d', function () {
        var a = tensor_util_env_1.convertToTensor([[[[1]], [[2]]], [[[3]], [[4]]]], 'a', 'test');
        expect(a.rank).toBe(4);
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        test_util_1.expectArraysClose(a, [1, 2, 3, 4]);
    });
    it('passing a tensor returns the tensor itself', function () {
        var s = tf.scalar(3);
        var res = tensor_util_env_1.convertToTensor(s, 'a', 'test');
        expect(res).toBe(s);
    });
    it('passing a tensor with casting returns the tensor itself', function () {
        var s = tf.scalar(3);
        var res = tensor_util_env_1.convertToTensor(s, 'a', 'test', 'bool');
        expect(res).toBe(s);
    });
    it('fails to convert a dict to tensor', function () {
        expect(function () { return tensor_util_env_1.convertToTensor({}, 'a', 'test'); })
            .toThrowError('Argument \'a\' passed to \'test\' must be a Tensor ' +
            'or TensorLike, but got Object');
    });
    it('fails to convert a string to tensor', function () {
        expect(function () { return tensor_util_env_1.convertToTensor('asdf', 'a', 'test'); })
            .toThrowError('Argument \'a\' passed to \'test\' must be a Tensor ' +
            'or TensorLike, but got String');
    });
    it('fails to convert a non-valid shape array to tensor', function () {
        var a = [[1, 2], [3], [4, 5, 6]];
        expect(function () { return tensor_util_env_1.convertToTensor(a, 'a', 'test'); })
            .toThrowError('Element arr[1] should have 2 elements, but has 1 elements');
    });
});
//# sourceMappingURL=tensor_util_test.js.map