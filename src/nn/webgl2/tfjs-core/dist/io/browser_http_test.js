"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var browser_http_1 = require("./browser_http");
var modelTopology1 = {
    'class_name': 'Sequential',
    'keras_version': '2.1.4',
    'config': [{
            'class_name': 'Dense',
            'config': {
                'kernel_initializer': {
                    'class_name': 'VarianceScaling',
                    'config': {
                        'distribution': 'uniform',
                        'scale': 1.0,
                        'seed': null,
                        'mode': 'fan_avg'
                    }
                },
                'name': 'dense',
                'kernel_constraint': null,
                'bias_regularizer': null,
                'bias_constraint': null,
                'dtype': 'float32',
                'activation': 'linear',
                'trainable': true,
                'kernel_regularizer': null,
                'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                'units': 1,
                'batch_input_shape': [null, 3],
                'use_bias': true,
                'activity_regularizer': null
            }
        }],
    'backend': 'tensorflow'
};
jasmine_util_1.describeWithFlags('browserHTTPRequest-load fetch-polyfill', test_util_1.NODE_ENVS, function () {
    var requestInits;
    beforeEach(function () {
        global.fetch = function () { };
        requestInits = [];
    });
    afterAll(function () {
        delete global.fetch;
    });
    var fakeResponse = function (body) { return ({
        ok: true,
        json: function () {
            return Promise.resolve(JSON.parse(body));
        },
        arrayBuffer: function () {
            var buf = body.buffer ?
                body.buffer :
                body;
            return Promise.resolve(buf);
        }
    }); };
    var setupFakeWeightFiles = function (fileBufferMap) {
        spyOn(global, 'fetch')
            .and.callFake(function (path, init) {
            requestInits.push(init);
            return fakeResponse(fileBufferMap[path]);
        });
    };
    it('1 group, 2 weights, 1 path', function (done) {
        var weightManifest1 = [{
                paths: ['weightfile0'],
                weights: [
                    {
                        name: 'dense/kernel',
                        shape: [3, 1],
                        dtype: 'float32',
                    },
                    {
                        name: 'dense/bias',
                        shape: [2],
                        dtype: 'float32',
                    }
                ]
            }];
        var floatData = new Float32Array([1, 3, 3, 7, 4]);
        setupFakeWeightFiles({
            './model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightManifest1 }),
            './weightfile0': floatData,
        });
        var handler = tf.io.browserHTTPRequest('./model.json');
        handler.load()
            .then(function (modelArtifacts) {
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(floatData);
            expect(requestInits).toEqual([{}, {}]);
            done();
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('throw exception if no fetch polyfill', function () {
        delete global.fetch;
        try {
            tf.io.browserHTTPRequest('./model.json');
        }
        catch (err) {
            expect(err.message)
                .toMatch(/not supported outside the web browser without a fetch polyfill/);
        }
    });
});
jasmine_util_1.describeWithFlags('browserHTTPRequest-save', test_util_1.CHROME_ENVS, function () {
    var weightSpecs1 = [
        {
            name: 'dense/kernel',
            shape: [3, 1],
            dtype: 'float32',
        },
        {
            name: 'dense/bias',
            shape: [1],
            dtype: 'float32',
        }
    ];
    var weightData1 = new ArrayBuffer(16);
    var artifacts1 = {
        modelTopology: modelTopology1,
        weightSpecs: weightSpecs1,
        weightData: weightData1,
    };
    var requestInits = [];
    beforeEach(function () {
        requestInits = [];
        spyOn(window, 'fetch').and.callFake(function (path, init) {
            if (path === 'model-upload-test' || path === 'http://model-upload-test') {
                requestInits.push(init);
                return new Response(null, { status: 200 });
            }
            else {
                return new Response(null, { status: 404 });
            }
        });
    });
    it('Save topology and weights, default POST method', function (done) {
        var testStartDate = new Date();
        var handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
        handler.save(artifacts1)
            .then(function (saveResult) {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
                .toEqual(JSON.stringify(weightSpecs1).length);
            expect(saveResult.modelArtifactsInfo.weightDataBytes)
                .toEqual(weightData1.byteLength);
            expect(requestInits.length).toEqual(1);
            var init = requestInits[0];
            expect(init.method).toEqual('POST');
            var body = init.body;
            var jsonFile = body.get('model.json');
            var jsonFileReader = new FileReader();
            jsonFileReader.onload = function (event) {
                var modelJSON = JSON.parse(event.target.result);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                expect(modelJSON.weightsManifest.length).toEqual(1);
                expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);
                var weightsFile = body.get('model.weights.bin');
                var weightsFileReader = new FileReader();
                weightsFileReader.onload = function (event) {
                    var weightData = event.target.result;
                    expect(new Uint8Array(weightData))
                        .toEqual(new Uint8Array(weightData1));
                    done();
                };
                weightsFileReader.onerror = function (error) {
                    done.fail(error.target.error.message);
                };
                weightsFileReader.readAsArrayBuffer(weightsFile);
            };
            jsonFileReader.onerror = function (error) {
                done.fail(error.target.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
    it('Save topology only, default POST method', function (done) {
        var testStartDate = new Date();
        var handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
        var topologyOnlyArtifacts = { modelTopology: modelTopology1 };
        handler.save(topologyOnlyArtifacts)
            .then(function (saveResult) {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes).toEqual(0);
            expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(0);
            expect(requestInits.length).toEqual(1);
            var init = requestInits[0];
            expect(init.method).toEqual('POST');
            var body = init.body;
            var jsonFile = body.get('model.json');
            var jsonFileReader = new FileReader();
            jsonFileReader.onload = function (event) {
                var modelJSON = JSON.parse(event.target.result);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                expect(body.get('model.weights.bin')).toEqual(null);
                done();
            };
            jsonFileReader.onerror = function (error) {
                done.fail(error.target.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
    it('Save topology and weights, PUT method, extra headers', function (done) {
        var testStartDate = new Date();
        var handler = tf.io.browserHTTPRequest('model-upload-test', {
            method: 'PUT',
            headers: { 'header_key_1': 'header_value_1', 'header_key_2': 'header_value_2' }
        });
        handler.save(artifacts1)
            .then(function (saveResult) {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
                .toEqual(JSON.stringify(weightSpecs1).length);
            expect(saveResult.modelArtifactsInfo.weightDataBytes)
                .toEqual(weightData1.byteLength);
            expect(requestInits.length).toEqual(1);
            var init = requestInits[0];
            expect(init.method).toEqual('PUT');
            expect(init.headers).toEqual({
                'header_key_1': 'header_value_1',
                'header_key_2': 'header_value_2'
            });
            var body = init.body;
            var jsonFile = body.get('model.json');
            var jsonFileReader = new FileReader();
            jsonFileReader.onload = function (event) {
                var modelJSON = JSON.parse(event.target.result);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                expect(modelJSON.weightsManifest.length).toEqual(1);
                expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);
                var weightsFile = body.get('model.weights.bin');
                var weightsFileReader = new FileReader();
                weightsFileReader.onload = function (event) {
                    var weightData = event.target.result;
                    expect(new Uint8Array(weightData))
                        .toEqual(new Uint8Array(weightData1));
                    done();
                };
                weightsFileReader.onerror = function (error) {
                    done.fail(error.target.error.message);
                };
                weightsFileReader.readAsArrayBuffer(weightsFile);
            };
            jsonFileReader.onerror = function (error) {
                done.fail(error.target.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
    it('404 response causes Error', function (done) {
        var handler = tf.io.getSaveHandlers('http://invalid/path')[0];
        handler.save(artifacts1)
            .then(function (saveResult) {
            done.fail('Calling browserHTTPRequest at invalid URL succeeded ' +
                'unexpectedly');
        })
            .catch(function (err) {
            done();
        });
    });
    it('getLoadHandlers with one URL string', function () {
        var handlers = tf.io.getLoadHandlers('http://foo/model.json');
        expect(handlers.length).toEqual(1);
        expect(handlers[0] instanceof browser_http_1.BrowserHTTPRequest).toEqual(true);
    });
    it('getLoadHandlers with two URL strings', function () {
        var handlers = tf.io.getLoadHandlers(['https://foo/graph.pb', 'https://foo/weights_manifest.json']);
        expect(handlers.length).toEqual(1);
        expect(handlers[0] instanceof browser_http_1.BrowserHTTPRequest).toEqual(true);
    });
    it('Existing body leads to Error', function () {
        expect(function () { return tf.io.browserHTTPRequest('model-upload-test', {
            body: 'existing body'
        }); }).toThrowError(/requestInit is expected to have no pre-existing body/);
    });
    it('Empty, null or undefined URL paths lead to Error', function () {
        expect(function () { return tf.io.browserHTTPRequest(null); })
            .toThrowError(/must not be null, undefined or empty/);
        expect(function () { return tf.io.browserHTTPRequest(undefined); })
            .toThrowError(/must not be null, undefined or empty/);
        expect(function () { return tf.io.browserHTTPRequest(''); })
            .toThrowError(/must not be null, undefined or empty/);
    });
    it('router', function () {
        expect(browser_http_1.httpRequestRouter('http://bar/foo') instanceof browser_http_1.BrowserHTTPRequest)
            .toEqual(true);
        expect(browser_http_1.httpRequestRouter('https://localhost:5000/upload') instanceof
            browser_http_1.BrowserHTTPRequest)
            .toEqual(true);
        expect(browser_http_1.httpRequestRouter('localhost://foo')).toBeNull();
        expect(browser_http_1.httpRequestRouter('foo:5000/bar')).toBeNull();
    });
});
jasmine_util_1.describeWithFlags('parseUrl', test_util_1.BROWSER_ENVS, function () {
    it('should parse url with no suffix', function () {
        var url = 'http://google.com/file';
        var _a = browser_http_1.parseUrl(url), prefix = _a[0], suffix = _a[1];
        expect(prefix).toEqual('http://google.com/');
        expect(suffix).toEqual('');
    });
    it('should parse url with suffix', function () {
        var url = 'http://google.com/file?param=1';
        var _a = browser_http_1.parseUrl(url), prefix = _a[0], suffix = _a[1];
        expect(prefix).toEqual('http://google.com/');
        expect(suffix).toEqual('?param=1');
    });
    it('should parse url with multiple serach params', function () {
        var url = 'http://google.com/a?x=1/file?param=1';
        var _a = browser_http_1.parseUrl(url), prefix = _a[0], suffix = _a[1];
        expect(prefix).toEqual('http://google.com/a?x=1/');
        expect(suffix).toEqual('?param=1');
    });
});
jasmine_util_1.describeWithFlags('browserHTTPRequest-load', test_util_1.BROWSER_ENVS, function () {
    describe('JSON model', function () {
        var requestInits;
        var setupFakeWeightFiles = function (fileBufferMap) {
            spyOn(window, 'fetch').and.callFake(function (path, init) {
                requestInits.push(init);
                return new Response(fileBufferMap[path]);
            });
        };
        beforeEach(function () {
            requestInits = [];
        });
        it('1 group, 2 weights, 1 path', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightManifest1 }),
                './weightfile0': floatData,
            });
            var handler = tf.io.browserHTTPRequest('./model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(floatData);
                expect(requestInits).toEqual([{}, {}]);
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('1 group, 2 weights, 1 path, with requestInit', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightManifest1 }),
                './weightfile0': floatData,
            });
            var handler = tf.io.browserHTTPRequest('./model.json', { headers: { 'header_key_1': 'header_value_1' } });
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(floatData);
                expect(requestInits).toEqual([
                    { headers: { 'header_key_1': 'header_value_1' } },
                    { headers: { 'header_key_1': 'header_value_1' } }
                ]);
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('1 group, 2 weight, 2 paths', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0', 'weightfile1'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData1 = new Float32Array([1, 3, 3]);
            var floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightManifest1 }),
                './weightfile0': floatData1,
                './weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest('./model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(new Float32Array([1, 3, 3, 7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('2 groups, 2 weight, 2 paths', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }],
                }
            ];
            var floatData1 = new Float32Array([1, 3, 3]);
            var floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightsManifest }),
                './weightfile0': floatData1,
                './weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest('./model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(new Float32Array([1, 3, 3, 7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'bool',
                        }],
                }
            ];
            var floatData1 = new Int32Array([1, 3, 3]);
            var floatData2 = new Uint8Array([7, 4]);
            setupFakeWeightFiles({
                'path1/model.json': JSON.stringify({ modelTopology: modelTopology1, weightsManifest: weightsManifest }),
                'path1/weightfile0': floatData1,
                'path1/weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest('path1/model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                    .toEqual(new Int32Array([1, 3, 3]));
                expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                    .toEqual(new Uint8Array([7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('topology only', function (done) {
            setupFakeWeightFiles({
                './model.json': JSON.stringify({ modelTopology: modelTopology1 }),
            });
            var handler = tf.io.browserHTTPRequest('./model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
                expect(modelArtifacts.weightSpecs).toBeUndefined();
                expect(modelArtifacts.weightData).toBeUndefined();
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('weights only', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'float32',
                        }],
                }
            ];
            var floatData1 = new Int32Array([1, 3, 3]);
            var floatData2 = new Float32Array([-7, -4]);
            setupFakeWeightFiles({
                'path1/model.json': JSON.stringify({ weightsManifest: weightsManifest }),
                'path1/weightfile0': floatData1,
                'path1/weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest('path1/model.json');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toBeUndefined();
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                    .toEqual(new Int32Array([1, 3, 3]));
                expect(new Float32Array(modelArtifacts.weightData.slice(12, 20)))
                    .toEqual(new Float32Array([-7, -4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('Missing modelTopology and weightsManifest leads to error', function (done) {
            setupFakeWeightFiles({ 'path1/model.json': JSON.stringify({}) });
            var handler = tf.io.browserHTTPRequest('path1/model.json');
            handler.load()
                .then(function (modelTopology1) {
                done.fail('Loading from missing modelTopology and weightsManifest ' +
                    'succeeded expectedly.');
            })
                .catch(function (err) {
                expect(err.message)
                    .toMatch(/contains neither model topology or manifest/);
                done();
            });
        });
    });
    describe('Binary model', function () {
        var requestInits;
        var modelData;
        var setupFakeWeightFiles = function (fileBufferMap) {
            spyOn(window, 'fetch').and.callFake(function (path, init) {
                requestInits.push(init);
                return new Response(fileBufferMap[path]);
            });
        };
        beforeEach(function () {
            requestInits = [];
            modelData = new ArrayBuffer(5);
        });
        it('1 group, 2 weights, 1 path', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.pb': modelData,
                './weights_manifest.json': JSON.stringify(weightManifest1),
                './weightfile0': floatData,
            });
            var handler = tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(floatData);
                expect(requestInits).toEqual([{}, {}, {}]);
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('1 group, 2 weights, 1 path with suffix', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.pb?tfjs-format=file': modelData,
                './weights_manifest.json?tfjs-format=file': JSON.stringify(weightManifest1),
                './weightfile0?tfjs-format=file': floatData,
            });
            var handler = tf.io.browserHTTPRequest([
                './model.pb?tfjs-format=file',
                './weights_manifest.json?tfjs-format=file'
            ]);
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(floatData);
                expect(requestInits).toEqual([{}, {}, {}]);
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('1 group, 2 weights, 1 path, with requestInit', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.pb': modelData,
                './weights_manifest.json': JSON.stringify(weightManifest1),
                './weightfile0': floatData,
            });
            var handler = tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json'], { headers: { 'header_key_1': 'header_value_1' } });
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(floatData);
                expect(requestInits).toEqual([
                    { headers: { 'header_key_1': 'header_value_1' } },
                    { headers: { 'header_key_1': 'header_value_1' } },
                    { headers: { 'header_key_1': 'header_value_1' } },
                ]);
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('1 group, 2 weight, 2 paths', function (done) {
            var weightManifest1 = [{
                    paths: ['weightfile0', 'weightfile1'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            var floatData1 = new Float32Array([1, 3, 3]);
            var floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.pb': modelData,
                './weights_manifest.json': JSON.stringify(weightManifest1),
                './weightfile0': floatData1,
                './weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightManifest1[0].weights);
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(new Float32Array([1, 3, 3, 7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('2 groups, 2 weight, 2 paths', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }],
                }
            ];
            var floatData1 = new Float32Array([1, 3, 3]);
            var floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.pb': modelData,
                './weights_manifest.json': JSON.stringify(weightsManifest),
                './weightfile0': floatData1,
                './weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest(['./model.pb', './weights_manifest.json']);
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Float32Array(modelArtifacts.weightData))
                    .toEqual(new Float32Array([1, 3, 3, 7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'bool',
                        }],
                }
            ];
            var floatData1 = new Int32Array([1, 3, 3]);
            var floatData2 = new Uint8Array([7, 4]);
            setupFakeWeightFiles({
                'path1/model.pb': modelData,
                'path2/weights_manifest.json': JSON.stringify(weightsManifest),
                'path2/weightfile0': floatData1,
                'path2/weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest(['path1/model.pb', 'path2/weights_manifest.json']);
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                    .toEqual(new Int32Array([1, 3, 3]));
                expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                    .toEqual(new Uint8Array([7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('2 groups, 2 weight, weight path prefix, Int32 and Uint8 Data', function (done) {
            var weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'bool',
                        }],
                }
            ];
            var floatData1 = new Int32Array([1, 3, 3]);
            var floatData2 = new Uint8Array([7, 4]);
            setupFakeWeightFiles({
                'path1/model.pb': modelData,
                'path2/weights_manifest.json': JSON.stringify(weightsManifest),
                'path3/weightfile0': floatData1,
                'path3/weightfile1': floatData2,
            });
            var handler = tf.io.browserHTTPRequest(['path1/model.pb', 'path2/weights_manifest.json'], {}, 'path3/');
            handler.load()
                .then(function (modelArtifacts) {
                expect(modelArtifacts.modelTopology).toEqual(modelData);
                expect(modelArtifacts.weightSpecs)
                    .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
                expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                    .toEqual(new Int32Array([1, 3, 3]));
                expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                    .toEqual(new Uint8Array([7, 4]));
                done();
            })
                .catch(function (err) { return done.fail(err.stack); });
        });
        it('the url path length is not 2 should leads to error', function () {
            expect(function () { return tf.io.browserHTTPRequest(['path1/model.pb']); }).toThrow();
        });
    });
});
//# sourceMappingURL=browser_http_test.js.map