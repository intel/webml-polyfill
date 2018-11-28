"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
jasmine_util_1.describeWithFlags('multinomial', test_util_1.ALL_ENVS, function () {
    var NUM_SAMPLES = 1000;
    var EPSILON = 0.05;
    var SEED = 3.14;
    it('Flip a fair coin and check bounds', function () {
        var probs = tf.tensor1d([1, 1]);
        var result = tf.multinomial(probs, NUM_SAMPLES, SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util_1.expectArraysClose(outcomeProbs, [0.5, 0.5], EPSILON);
    });
    it('Flip a two-sided coin with 100% of heads', function () {
        var logits = tf.tensor1d([1, -100]);
        var result = tf.multinomial(logits, NUM_SAMPLES, SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util_1.expectArraysClose(outcomeProbs, [1, 0], EPSILON);
    });
    it('Flip a two-sided coin with 100% of tails', function () {
        var logits = tf.tensor1d([-100, 1]);
        var result = tf.multinomial(logits, NUM_SAMPLES, SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), 2);
        test_util_1.expectArraysClose(outcomeProbs, [0, 1], EPSILON);
    });
    it('Flip a single-sided coin throws error', function () {
        var probs = tf.tensor1d([1]);
        expect(function () { return tf.multinomial(probs, NUM_SAMPLES, SEED); }).toThrowError();
    });
    it('Flip a ten-sided coin and check bounds', function () {
        var numOutcomes = 10;
        var logits = tf.fill([numOutcomes], 1).as1D();
        var result = tf.multinomial(logits, NUM_SAMPLES, SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync(), numOutcomes);
        expect(outcomeProbs.length).toBeLessThanOrEqual(numOutcomes);
    });
    it('Flip 3 three-sided coins, each coin is 100% biases', function () {
        var numOutcomes = 3;
        var logits = tf.tensor2d([[-100, -100, 1], [-100, 1, -100], [1, -100, -100]], [3, numOutcomes]);
        var result = tf.multinomial(logits, NUM_SAMPLES, SEED);
        expect(result.dtype).toBe('int32');
        expect(result.shape).toEqual([3, NUM_SAMPLES]);
        var outcomeProbs = computeProbs(result.dataSync().slice(0, NUM_SAMPLES), numOutcomes);
        test_util_1.expectArraysClose(outcomeProbs, [0, 0, 1], EPSILON);
        outcomeProbs = computeProbs(result.dataSync().slice(NUM_SAMPLES, 2 * NUM_SAMPLES), numOutcomes);
        test_util_1.expectArraysClose(outcomeProbs, [0, 1, 0], EPSILON);
        outcomeProbs =
            computeProbs(result.dataSync().slice(2 * NUM_SAMPLES), numOutcomes);
        test_util_1.expectArraysClose(outcomeProbs, [1, 0, 0], EPSILON);
    });
    it('passing Tensor3D throws error', function () {
        var probs = tf.zeros([3, 2, 2]);
        var normalized = true;
        expect(function () { return tf.multinomial(probs, 3, SEED, normalized); })
            .toThrowError();
    });
    it('throws when passed a non-tensor', function () {
        expect(function () { return tf.multinomial({}, NUM_SAMPLES, SEED); })
            .toThrowError(/Argument 'logits' passed to 'multinomial' must be a Tensor/);
    });
    it('accepts a tensor-like object for logits (biased coin)', function () {
        var res = tf.multinomial([-100, 1], NUM_SAMPLES, SEED);
        expect(res.dtype).toBe('int32');
        expect(res.shape).toEqual([NUM_SAMPLES]);
        var outcomeProbs = computeProbs(res.dataSync(), 2);
        test_util_1.expectArraysClose(outcomeProbs, [0, 1], EPSILON);
    });
    function computeProbs(events, numOutcomes) {
        var counts = [];
        for (var i = 0; i < numOutcomes; ++i) {
            counts[i] = 0;
        }
        var numSamples = events.length;
        for (var i = 0; i < events.length; ++i) {
            counts[events[i]]++;
        }
        for (var i = 0; i < counts.length; i++) {
            counts[i] /= numSamples;
        }
        return counts;
    }
});
//# sourceMappingURL=multinomial_test.js.map