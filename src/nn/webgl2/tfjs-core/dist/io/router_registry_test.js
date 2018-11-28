"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("../index");
var jasmine_util_1 = require("../jasmine_util");
var test_util_1 = require("../test_util");
var indexed_db_1 = require("./indexed_db");
var local_storage_1 = require("./local_storage");
var router_registry_1 = require("./router_registry");
jasmine_util_1.describeWithFlags('IORouterRegistry', test_util_1.BROWSER_ENVS, function () {
    var localStorageRouter = function (url) {
        var scheme = 'localstorage://';
        if (url.startsWith(scheme)) {
            return local_storage_1.browserLocalStorage(url.slice(scheme.length));
        }
        else {
            return null;
        }
    };
    var indexedDBRouter = function (url) {
        var scheme = 'indexeddb://';
        if (url.startsWith(scheme)) {
            return indexed_db_1.browserIndexedDB(url.slice(scheme.length));
        }
        else {
            return null;
        }
    };
    var FakeIOHandler = (function () {
        function FakeIOHandler(url1, url2) {
        }
        return FakeIOHandler;
    }());
    var fakeMultiStringRouter = function (url) {
        var scheme = 'foo://';
        if (Array.isArray(url) && url.length === 2) {
            if (url[0].startsWith(scheme) && url[1].startsWith(scheme)) {
                return new FakeIOHandler(url[0], url[1]);
            }
            else {
                return null;
            }
        }
        else {
            return null;
        }
    };
    var tempRegistryInstance = null;
    beforeEach(function () {
        tempRegistryInstance = router_registry_1.IORouterRegistry.instance;
        router_registry_1.IORouterRegistry.instance = null;
    });
    afterEach(function () {
        router_registry_1.IORouterRegistry.instance = tempRegistryInstance;
    });
    it('getSaveHandler succeeds', function () {
        router_registry_1.IORouterRegistry.registerSaveRouter(localStorageRouter);
        router_registry_1.IORouterRegistry.registerSaveRouter(indexedDBRouter);
        var out1 = tf.io.getSaveHandlers('localstorage://foo-model');
        expect(out1.length).toEqual(1);
        expect(out1[0] instanceof local_storage_1.BrowserLocalStorage).toEqual(true);
        var out2 = tf.io.getSaveHandlers('indexeddb://foo-model');
        expect(out2.length).toEqual(1);
        expect(out2[0] instanceof indexed_db_1.BrowserIndexedDB).toEqual(true);
    });
    it('getLoadHandler succeeds', function () {
        router_registry_1.IORouterRegistry.registerLoadRouter(localStorageRouter);
        router_registry_1.IORouterRegistry.registerLoadRouter(indexedDBRouter);
        var out1 = tf.io.getLoadHandlers('localstorage://foo-model');
        expect(out1.length).toEqual(1);
        expect(out1[0] instanceof local_storage_1.BrowserLocalStorage).toEqual(true);
        var out2 = tf.io.getLoadHandlers('indexeddb://foo-model');
        expect(out2.length).toEqual(1);
        expect(out2[0] instanceof indexed_db_1.BrowserIndexedDB).toEqual(true);
    });
    it('getLoadHandler with string array argument succeeds', function () {
        router_registry_1.IORouterRegistry.registerLoadRouter(fakeMultiStringRouter);
        var loadHandler = router_registry_1.IORouterRegistry.getLoadHandlers(['foo:///123', 'foo:///456']);
        expect(loadHandler[0] instanceof FakeIOHandler).toEqual(true);
        expect(router_registry_1.IORouterRegistry.getLoadHandlers(['foo:///123', 'bar:///456']))
            .toEqual([]);
        expect(router_registry_1.IORouterRegistry.getLoadHandlers(['foo:///123'])).toEqual([]);
        expect(router_registry_1.IORouterRegistry.getLoadHandlers('foo:///123')).toEqual([]);
    });
    it('getSaveHandler fails', function () {
        router_registry_1.IORouterRegistry.registerSaveRouter(localStorageRouter);
        expect(tf.io.getSaveHandlers('invalidscheme://foo-model')).toEqual([]);
        expect(tf.io.getLoadHandlers('localstorage://foo-model')).toEqual([]);
    });
});
//# sourceMappingURL=router_registry_test.js.map