"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("./index");
var jasmine_util_1 = require("./jasmine_util");
var tape_1 = require("./tape");
var test_util_1 = require("./test_util");
jasmine_util_1.describeWithFlags('getFilteredNodesXToY', test_util_1.ALL_ENVS, function () {
    it('no paths from x to y', function () {
        var x = tf.scalar(1);
        var intermediate1 = tf.scalar(0);
        var intermediate2 = tf.scalar(0);
        var y = tf.scalar(2);
        var tape = [
            {
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [intermediate1],
                gradient: null
            },
            {
                id: 1,
                name: 'node1',
                inputs: { intermediate2: intermediate2 },
                outputs: [y],
                gradient: null
            }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x], y);
        expect(filteredTapeNodes.length).toBe(0);
        expect(filteredTapeNodes).toEqual([]);
    });
    it('one operation x => y', function () {
        var x = tf.scalar(1);
        var y = tf.scalar(2);
        var tape = [{ id: 0, name: 'node0', inputs: { x: x }, outputs: [y], gradient: null }];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x], y);
        expect(filteredTapeNodes.length).toBe(1);
        expect(filteredTapeNodes).toEqual(tape);
    });
    it('1 operation [x0, x1] => y, all input paths', function () {
        var x0 = tf.scalar(0);
        var x1 = tf.scalar(1);
        var y = tf.scalar(2);
        var tape = [
            { id: 0, name: 'node0', inputs: { x0: x0, x1: x1 }, outputs: [y], gradient: null }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x0, x1], y);
        expect(filteredTapeNodes.length).toBe(1);
        expect(filteredTapeNodes).toEqual(tape);
    });
    it('one operation [x0, x1] => y, one input paths', function () {
        var x0 = tf.scalar(0);
        var x1 = tf.scalar(1);
        var y = tf.scalar(2);
        var tape = [
            { id: 0, name: 'node0', inputs: { x0: x0, x1: x1 }, outputs: [y], gradient: null }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x0], y);
        expect(filteredTapeNodes.length).toBe(1);
        expect(filteredTapeNodes[0])
            .toEqual({ id: 0, name: 'node0', inputs: { x0: x0 }, outputs: [y], gradient: null });
    });
    it('two operations x => intermediate => y', function () {
        var x = tf.scalar(1);
        var intermediate = tf.scalar(0);
        var y = tf.scalar(2);
        var tape = [
            {
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [intermediate],
                gradient: null
            },
            {
                id: 1,
                name: 'node1',
                inputs: { intermediate: intermediate },
                outputs: [y],
                gradient: null
            }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x], y);
        expect(filteredTapeNodes.length).toBe(2);
        expect(filteredTapeNodes).toEqual(tape);
    });
    it('two operations [x0, x1], [x2] => ' +
        'intermediate => y', function () {
        var x0 = tf.scalar(1);
        var x1 = tf.scalar(2);
        var x2 = tf.scalar(3);
        var intermediate = tf.scalar(4);
        var y = tf.scalar(2);
        var tape = [
            {
                id: 0,
                name: 'node0',
                inputs: { x0: x0, x1: x1 },
                outputs: [intermediate],
                gradient: null
            },
            {
                id: 1,
                name: 'node1',
                inputs: { x2: x2, intermediate: intermediate },
                outputs: [y],
                gradient: null
            }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x0, x1, x2], y);
        expect(filteredTapeNodes.length).toBe(2);
        expect(filteredTapeNodes).toEqual(tape);
    });
    it('x => y and x => orphan', function () {
        var x = tf.scalar(1);
        var orphan = tf.scalar(0);
        var y = tf.scalar(2);
        var tape = [
            { id: 0, name: 'node0', inputs: { x: x }, outputs: [orphan], gradient: null },
            { id: 1, name: 'node1', inputs: { x: x }, outputs: [y], gradient: null }
        ];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x], y);
        expect(filteredTapeNodes.length).toBe(1);
        expect(filteredTapeNodes[0]).toEqual(tape[1]);
    });
    it('x => y and orphan => y', function () {
        var x = tf.scalar(1);
        var orphan = tf.scalar(0);
        var y = tf.scalar(2);
        var tape = [{
                id: 0,
                name: 'node0',
                inputs: { x: x, orphan: orphan },
                outputs: [y],
                gradient: null
            }];
        var filteredTapeNodes = tape_1.getFilteredNodesXToY(tape, [x], y);
        expect(filteredTapeNodes.length).toBe(1);
        expect(filteredTapeNodes[0])
            .toEqual({ id: 0, name: 'node0', inputs: { x: x }, outputs: [y], gradient: null });
    });
    it('1 op with 3 outputs x => y1, y2, y3', function () {
        var x = tf.scalar(1);
        var y1 = tf.scalar(2);
        var y2 = tf.scalar(2);
        var y3 = tf.scalar(2);
        var tape = [{
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [y1, y2, y3],
                gradient: null
            }];
        var filteredNodes1 = tape_1.getFilteredNodesXToY(tape, [x], y1);
        expect(filteredNodes1.length).toBe(1);
        expect(filteredNodes1).toEqual(tape);
        var filteredNodes2 = tape_1.getFilteredNodesXToY(tape, [x], y2);
        expect(filteredNodes2.length).toBe(1);
        expect(filteredNodes2).toEqual(tape);
        var filteredNodes3 = tape_1.getFilteredNodesXToY(tape, [x], y3);
        expect(filteredNodes3.length).toBe(1);
        expect(filteredNodes3).toEqual(tape);
    });
});
jasmine_util_1.describeWithFlags('backpropagateGradients', test_util_1.ALL_ENVS, function () {
    it('Throws if gradient is not defined', function () {
        var x = tf.scalar(0);
        var y = tf.scalar(1);
        var dy = tf.scalar(1);
        var accumulatedGradientsMap = {};
        accumulatedGradientsMap[y.id] = dy;
        var tape = [{ id: 0, name: 'node0', inputs: { x: x }, outputs: [y], gradient: null }];
        expect(function () { return tape_1.backpropagateGradients(accumulatedGradientsMap, tape); })
            .toThrowError();
    });
    it('basic backprop with 1 node', function () {
        var x = tf.scalar(0);
        var y = tf.scalar(1);
        var dy = tf.scalar(1);
        var accumulatedGradientsMap = {};
        accumulatedGradientsMap[y.id] = dy;
        var tape = [{
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [y],
                gradient: function (dy) {
                    return { x: function () { return dy.add(tf.scalar(1)); } };
                }
            }];
        tape_1.backpropagateGradients(accumulatedGradientsMap, tape);
        test_util_1.expectArraysClose(accumulatedGradientsMap[x.id], [2]);
    });
    it('basic backprop with 2 nodes', function () {
        var x = tf.scalar(0);
        var intermediate = tf.scalar(1);
        var y = tf.scalar(2);
        var dy = tf.scalar(1);
        var accumulatedGradientsMap = {};
        accumulatedGradientsMap[y.id] = dy;
        var tape = [
            {
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [intermediate],
                gradient: function (dy) {
                    return { x: function () { return dy.add(tf.scalar(1)); } };
                }
            },
            {
                id: 1,
                name: 'node1',
                inputs: { intermediate: intermediate },
                outputs: [y],
                gradient: function (dy) {
                    return { intermediate: function () { return dy.add(tf.scalar(1)); } };
                }
            }
        ];
        tape_1.backpropagateGradients(accumulatedGradientsMap, tape);
        test_util_1.expectArraysClose(accumulatedGradientsMap[x.id], [3]);
    });
    it('basic backprop with a split node accumulates gradients', function () {
        var x = tf.scalar(0);
        var intermediate1 = tf.scalar(1);
        var intermediate2 = tf.scalar(2);
        var y = tf.scalar(3);
        var dy = tf.scalar(1);
        var accumulatedGradientsMap = {};
        accumulatedGradientsMap[y.id] = dy;
        var tape = [
            {
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [intermediate1],
                gradient: function (dy) {
                    return { x: function () { return dy.add(tf.scalar(1)); } };
                }
            },
            {
                id: 1,
                name: 'node1',
                inputs: { x: x },
                outputs: [intermediate2],
                gradient: function (dy) {
                    return { x: function () { return dy.add(tf.scalar(1)); } };
                }
            },
            {
                id: 2,
                name: 'node2',
                inputs: { intermediate1: intermediate1, intermediate2: intermediate2 },
                outputs: [y],
                gradient: function (dy) {
                    return {
                        intermediate1: function () { return dy.add(tf.scalar(1)); },
                        intermediate2: function () { return dy.add(tf.scalar(1)); }
                    };
                }
            }
        ];
        tape_1.backpropagateGradients(accumulatedGradientsMap, tape);
        test_util_1.expectArraysClose(accumulatedGradientsMap[x.id], [dy.dataSync()[0] + 5]);
    });
    it('backprop over 1 node with 3 outputs, w.r.t to the 2nd output', function () {
        var x = tf.tensor1d([1, 1, 1]);
        var y1 = tf.scalar(1);
        var y2 = tf.scalar(1);
        var y3 = tf.scalar(1);
        var accumulatedGradientsMap = {};
        var dy2 = tf.scalar(5);
        accumulatedGradientsMap[y2.id] = dy2;
        var dys;
        var tape = [{
                id: 0,
                name: 'node0',
                inputs: { x: x },
                outputs: [y1, y2, y3],
                gradient: function (dys_) {
                    dys = dys_;
                    return { x: function () { return tf.stack(dys_); } };
                }
            }];
        tape_1.backpropagateGradients(accumulatedGradientsMap, tape);
        test_util_1.expectArraysClose(accumulatedGradientsMap[x.id], [0, 5, 0]);
        test_util_1.expectArraysClose(dys[0], [0]);
        test_util_1.expectArraysClose(dys[1], [5]);
        test_util_1.expectArraysClose(dys[2], [0]);
    });
});
//# sourceMappingURL=tape_test.js.map