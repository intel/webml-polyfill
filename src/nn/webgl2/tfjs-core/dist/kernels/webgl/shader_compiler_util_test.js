"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var jasmine_util_1 = require("../../jasmine_util");
var test_util_1 = require("../../test_util");
var shader_compiler_util_1 = require("./shader_compiler_util");
jasmine_util_1.describeWithFlags('shader compiler', test_util_1.WEBGL_ENVS, function () {
    it('dotify takes two arrays of coordinates and produces' +
        'the glsl that finds the dot product of those coordinates', function () {
        var coords1 = ['r', 'g', 'b', 'a'];
        var coords2 = ['x', 'y', 'z', 'w'];
        expect(shader_compiler_util_1.dotify(coords1, coords2))
            .toEqual('dot(vec4(r,g,b,a), vec4(x,y,z,w))');
    });
    it('dotify should split up arrays into increments of vec4s', function () {
        var coords1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        var coords2 = ['h', 'i', 'j', 'k', 'l', 'm', 'n'];
        expect(shader_compiler_util_1.dotify(coords1, coords2))
            .toEqual('dot(vec4(a,b,c,d), vec4(h,i,j,k))+dot(vec3(e,f,g), vec3(l,m,n))');
    });
    it('getLogicalCoordinatesFromFlatIndex produces glsl that takes' +
        'a flat index and finds its coordinates within that shape', function () {
        var coords = ['r', 'c', 'd'];
        var shape = [1, 2, 3];
        expect(shader_compiler_util_1.getLogicalCoordinatesFromFlatIndex(coords, shape))
            .toEqual('int r = index / 6; index -= r * 6;' +
            'int c = index / 3; int d = index - c * 3;');
    });
});
//# sourceMappingURL=shader_compiler_util_test.js.map