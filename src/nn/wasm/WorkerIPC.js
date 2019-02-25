//
// Call an exported function defined in worker script
//
//  1. `foo` has no arguments:
//     let ret = await worker.dispatch('foo');
//   or
//     let ret = await worker.foo();
//
//  2. `foo` has one argument `bar`
//     let ret = await worker.dispatch('foo', { args: [bar] });
//   or
//     let ret = await worker.foo(bar);
//
//  3. `foo` has one transferrable argument `bar` (no syntatic sugar)
//     let ret = await worker.dispatch('foo', {args: [bar], transferList: [bar.buffer]});
//     

export function getPreparedModelWorker() {
  let _worker = require('worker-loader?inline!./PreparedModelWorker')();
  let _handlers = {};
  let _handlerCounter = 0;
  _worker.onmessage = function (e) {
    const [id, fn, ret, error] = e.data;
    const fnId = fn + id;
    if (!error) {
      const resolveHandler = _handlers[fnId][0];
      resolveHandler(ret);
    } else {
      const err = new Error();
      [err.message, err.stack] = ret;
      const rejectHandler = _handlers[fnId][1];
      rejectHandler(err);
    }
    delete _handlers[fnId];
  };
  _worker.dispatch = function (fn, msg = {}) {
    return new Promise((resolve, reject) => {
      const id = _handlerCounter++;
      const args = msg.args || [];
      const transferList = msg.transferList || [];
      this.postMessage([id, fn, args], transferList);
      _handlers[fn + id] = [resolve, reject];
    });
  };

  let export_functions = require('./PreparedModelWorker').export_functions;
  for (let funcName in export_functions) {
    // wire the function up to the worker so that exported functions can be
    // directly called on the worker object
    _worker[funcName] = async function() {
      return await _worker.dispatch(funcName, { args: Array.from(arguments) });
    }
  }

  return _worker;
}

export function getPreparedModelMain() {
  let _worker = {};
  let export_functions = require('./PreparedModelWorker').export_functions;
  _worker.dispatch = async function (fn, msg = {}) {
    const args = msg.args || [];
    const ret = await export_functions[fn](...args);
    if (typeof ret !== 'undefined') {
      return ret.retVal
    }
  };

  for (let funcName in export_functions) {
    // wire the function up to the worker so that exported functions can be
    // directly called on the worker object 
    _worker[funcName] = async function() {
      return await _worker.dispatch(funcName, { args: Array.from(arguments) });
    }
  }

  return _worker;
}