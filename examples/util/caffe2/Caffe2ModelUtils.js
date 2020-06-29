class Caffe2ModelUtils {
  constructor(predictModel, initModel, isDNNL=false) {
    this._predict = predictModel;
    this._init = initModel;
    this._isDNNL = isDNNL;
    this._inputName = "";
    this._dataFormat = false;       // NCHW to NHWC

    this._initUtils();
    this._initMap = this._initModelHandler();
    this._predictMap = this._predictModelHandler();
  }

  getCaffe2Model () {
    return this._predictMap;
  }

  getCaffe2InitModel () {
    return this._initMap;
  }

  _initModelHandler () {
    let initTensorMap = [];

    for (let opIdx in this._init.op) {
      initTensorMap[opIdx] = [];
      let op = this._init.op[opIdx];
      initTensorMap[opIdx][op.output[0]] = [];

      if (op.output[0] != this._inputName) {
        for (let argIdx in op.arg) {
          let arg = op.arg[argIdx];
          initTensorMap[opIdx][op.output[0]][arg.name] = [];
          let data = this._checkArgData(arg);
          initTensorMap[opIdx][op.output[0]][arg.name]["type"] = data.type;
          initTensorMap[opIdx][op.output[0]][arg.name]["value"] = data.value;
        }

        // Data handle
        initTensorMap[opIdx][op.output[0]] = this._DataHandleforTensor(initTensorMap[opIdx][op.output[0]]);
      }
    }

    return initTensorMap;
  }

  _predictModelHandler() {
    let predictTensorMap = [];

    for (let opIdx in this._predict.op) {
      predictTensorMap[opIdx] = [];
      let op = this._predict.op[opIdx];

      // name, type, engine
      predictTensorMap[opIdx]["name"] = op.name;
      predictTensorMap[opIdx]["operator"] = op.type;
      predictTensorMap[opIdx]["engine"] = op.engine;

      // input
      predictTensorMap[opIdx]["input"] = [];
      for (let inputIdx in op.input) {
        predictTensorMap[opIdx]["input"][inputIdx] = [];
        let input = op.input[inputIdx];
        predictTensorMap[opIdx]["input"][inputIdx]["name"] = input;

        for (let inTmp of this._initMap) {
          if (inTmp.hasOwnProperty(input)) {
            for (let key in inTmp[input]) {
              predictTensorMap[opIdx]["input"][inputIdx][key] = inTmp[input][key];
            }
          }
        }
      }

      // output
      predictTensorMap[opIdx]["output"] = [];
      for (let outputIdx in op.output) {
        predictTensorMap[opIdx]["output"][outputIdx] = [];
        let output = op.output[outputIdx];
        predictTensorMap[opIdx]["output"][outputIdx]["name"] = output;
      }

      // arg
      predictTensorMap[opIdx]["arg"] = [];
      for (let argIdx in op.arg) {
        let arg = op.arg[argIdx];

        predictTensorMap[opIdx]["arg"][arg.name] = [];
        let data = this._checkArgData(arg);
        predictTensorMap[opIdx]["arg"][arg.name]["type"] = data.type;
        predictTensorMap[opIdx]["arg"][arg.name]["value"] = data.value;

        // Data handle
        if (arg.name == "order") {
          predictTensorMap[opIdx]["arg"][arg.name]["type"] = "str";
          let orderTmp = [];
          for (let val of predictTensorMap[opIdx]["arg"][arg.name]["value"]) {
            orderTmp.push(String.fromCharCode(val));
          }
          predictTensorMap[opIdx]["arg"][arg.name]["value"] = orderTmp.join("");
        } else {
          predictTensorMap[opIdx]["arg"][arg.name] = this._DataHandleforArg(predictTensorMap[opIdx]["arg"][arg.name]);
        }
      }
    }

    return predictTensorMap;
  }

  _initUtils() {
    this._inputName = this._predict.externalInput[0];

    if (this._checkDataFormat(this._init) || this._checkDataFormat(this._predict)) {
      this._dataFormat = true;
    }
  }

  _checkDataFormat(model) {
    for (let op of model.op) {
      for (let arg of op.arg) {
        if (arg.name == "order") {
          let data = this._checkArgData(arg);
          let orderTmp = [];
          for (let val of data.value) {
            orderTmp.push(String.fromCharCode(val));
          }
          let formatStr = orderTmp.join("");
          if (formatStr == "NCHW") {
            return true;
          };
        }
      }
    }
  }

  _checkArgData(arg) {
    for (let [key, val] of Object.entries(arg)) {
      if (key != "name" && key != "tensors" && key != "nets" && key != "qtensors") {
        if (val.length !== 0) {
          return this._pareData(val, key);
        }
      }
    }
  }

  _pareData(dataValue, dataType) {
    switch(dataType) {
      case "i":
      case "ints": {
        dataType = "int32";
      } break;
      case "f":
      case "floats": {
        dataType = "float32";
      } break;
      case "s": {
        dataType = "uint8";
        let dataTmp = [];
        let buf = new Uint8Array(dataValue);
        for (let value of buf.values()) {
          dataTmp.push(value);
        }
        dataValue = dataTmp;
      } break;
      default: {
        throw new Error(`${dataType} is not supported.`);
      }
    };

    return {"type": dataType, "value": dataValue};
  }

  _DataHandleforArg (arg) {
    let argValue = arg["value"];
    let N = argValue[0];
    let C = argValue[1];
    let H = argValue[2];
    let W = argValue[3];

    if (this._dataFormat && argValue.length === 4) {
      arg["value"] = [N, H, W, C];
    }

    return arg;
  }

  _DataHandleforTensor (tensor) {
    if (this._isDNNL) {
      tensor = this._uint8Toint8(tensor);
    }

    // For shape
    let shapeValue = tensor["shape"]["value"];
    let shapeType = tensor["shape"]["type"];
    let shapeCtor = this._TypetoArray(shapeType);
    let N = shapeValue[0];
    let C = shapeValue[1];
    let H = shapeValue[2];
    let W = shapeValue[3];
    let flag = shapeValue.length === 4 ? true : false;

    if (this._dataFormat && flag) {
      tensor["shape"]["value"] = new shapeCtor([N, H, W, C]);
    } else {
      tensor["shape"]["value"] = new shapeCtor(this._DatatoArray(shapeValue));
    }

    // For value
    let valuesValue = tensor["values"]["value"];
    let valuesType = tensor["values"]["type"];
    let valuesCtor = this._TypetoArray(valuesType);

    let valuesValueTmp;
    if (this._dataFormat && flag) {
      valuesValueTmp = new valuesCtor(valuesValue.length);
      for (let n = 0; n < N; ++n) {
        for (let c = 0; c < C; ++c) {
          for (let h = 0; h < H; ++h) {
            for (let w = 0; w < W; ++w) {
              valuesValueTmp[n*H*W*C + h*W*C + w*C + c] = valuesValue[n*C*H*W + c*H*W + h*W + w];
            }
          }
        }
      }
    } else {
      valuesValueTmp = new valuesCtor(this._DatatoArray(valuesValue));
    }

    tensor["values"]["value"] = valuesValueTmp;

    return tensor;
  }

  _uint8Toint8(tensor) {
    // uint8 => int8
    if (typeof tensor["values"] != "undefined" &&
        tensor["values"]["type"] == "uint8" &&
        typeof tensor["Y_zero_point"] != "undefined" &&
        tensor["Y_zero_point"]["value"] == "128") {
      tensor["values"]["type"] = "int8";
      let tmpArray = [];
      for ( let val of Object.values(tensor["values"]["value"])) {
        tmpArray.push(val - 128);
      }
      tensor["values"]["value"] = tmpArray;
      tensor["Y_zero_point"]["value"] = "0";
    }

    return tensor;
  }

  _TypetoArray (type) {
    let ctor;
    if (type == "int32") ctor = Int32Array;
    else if (type == "uint32") ctor = Uint32Array;
    else if (type == "float32") ctor = Float32Array;
    else if (type == "uint8") ctor = Uint8Array;
    else if (type == "int8") ctor = Int8Array;
    else if (type == "str") ctor = Array;
    else throw new Error(`${type} is not supported.`);
    return ctor;
  }

  _DatatoArray (data) {
    let dataTmp = [];
    if (typeof data.length == "undefined" || data.length == 1) {
      dataTmp.push(data);
    } else {
      dataTmp = data;
    }
    return dataTmp;
  }
}
