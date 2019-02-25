import PreparedModelWorker from 'worker-loader?inline!./PreparedModelWorker';

export function getPreparedModelWorker() {
  let _worker = new PreparedModelWorker();
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
      const rejectHandler = _handlers[fnId][0];
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
  return _worker;
}