class OpenVINOModel {
  /*
   * networkText: string
   * weightsBuffer: ArrayBuffer
  */
  constructor(networkText, weightsBuffer) {
    this._network = this._verifyAndParse(networkText);
    this._weights = weightsBuffer;

    const version = parseInt(this._network._graphs[0]._version);
    if (version < 5) {
      throw new Error(`IR version ${version} is not supported. ` +
          `Please convert the model using the latest OpenVINO model optimizer`);
    }

    this._bindHelperFunctions();
  }

  get network() {
    return this._network;
  }

  get weights() {
    return this._weights;
  }

  get graphs() {
    return this._network.graphs;
  }

  _verifyAndParse(networkText) {
    const parser = new DOMParser({ errorHandler: () => { errors = true; } });
    const xml = parser.parseFromString(networkText, 'text/xml');

    if (xml.documentElement == null ||
        xml.getElementsByTagName('parsererror').length > 0) {
      throw new openvino.Error('File format is not OpenVINO XML');
    }
    const net = xml.documentElement;
    if (!net || net.nodeName != 'net' ||
        openvino.Node.children(net, 'layers').length != 1 ||
        openvino.Node.children(net, 'edges').length != 1) {
      throw new openvino.Error('File format is not OpenVINO IR');
    }

    // don't use metadata
    const metadata = new openvino.Metadata(null);
    return new openvino.Model(metadata, net);
  }

  _bindHelperFunctions() {
    // collect all nodes and tensors in the model
    const allNodes = [];
    const allTensors = [];
    for (const graph of this._network.graphs) {
      for (const node of graph.nodes) {
        allNodes.push(node);
        allTensors.push(...node.inputs);
        allTensors.push(...node.outputs);
      }
      allTensors.push(...graph.inputs);
      allTensors.push(...graph.outputs);
    }

    // bind helper functions for openvino.Node
    for (const node of allNodes) {
      node.getBool = (name, defaultValue) => this.getBool(node, name, defaultValue);
      node.getInt = (name, defaultValue) => this.getInt(node, name, defaultValue);
      node.getInts = (name, defaultValue) => this.getInts(node, name, defaultValue);
      node.getFloat = (name, defaultValue) => this.getFloat(node, name, defaultValue);
      node.getFloats = (name, defaultValue) => this.getFloats(node, name, defaultValue);
      node.getString = (name, defaultValue) => this.getString(node, name, defaultValue);
      node.getInitializer = (dimHints) => this.getNodeInitilizer(node, dimHints);
      node.getKernelShape = () => this.getKernelShape(node);
    }

    // bind helper functions for openvino.Tensor
    for (const tensor of allTensors) {
      tensor.graphId = () => this.getTensorGraphId(tensor);
      tensor.dataType = () => this.getTensorDataType(tensor);
      tensor.shape = () => this.getTensorShape(tensor);
      tensor.getInitializer = (dimHints) => this.getTensorInitializer(tensor, dimHints);
    }
  }

  // Helper functions for openvino.Node
  _getObjectByName(array, name) {
    return array.filter((x) => x.name === name)[0];
  }

  _getAttributeValue(node, name, defaultValue) {
    const attribute = this._getObjectByName(node.attributes, name);
    if (typeof attribute === 'undefined') {
      return defaultValue;  // return undefined if not given a default value
    }
    return attribute.value;
  }

  getString(node, name, defaultValue) {
    return this._getAttributeValue(node, name, defaultValue);
  }

  getBool(node, name, defaultValue) {
    const attribute = this._getAttributeValue(node, name, defaultValue);
    if (attribute === 'true' || attribute === true) {
      return true;
    } else if (attribute === 'false' || attribute === false) {
      return false;
    } else {
      throw new Error('Not a boolean');
    }
  }

  getInt(node, name, defaultValue) {
    const attribute = this._getAttributeValue(node, name, defaultValue);
    if (typeof attribute !== 'undefined') {
      return parseInt(attribute);
    }
  }

  getInts(node, name, defaultValue) {
    const attribute = this._getAttributeValue(node, name, defaultValue);
    if (typeof attribute !== 'undefined') {
      return attribute.split(',').map((s) => parseInt(s));
    }
  }

  getFloat(node, name, defaultValue) {
    const attribute = this._getAttributeValue(node, name, defaultValue);
    if (typeof attribute !== 'undefined') {
      return parseFloat(attribute);
    }
  }

  getFloats(node, name, defaultValue) {
    const attribute = this._getAttributeValue(node, name, defaultValue);
    if (typeof attribute !== 'undefined') {
      return attribute.split(',').map((s) => parseFloat(s));
    }
  }

  getKernelShape(node) {
    if (node.operator === 'Convolution') {
      const [kernelH, kernelW] = this.getInts(node, 'kernel');
      const inputDims = this.getTensorShape(node.inputs[0]);
      const outputDims = this.getTensorShape(node.outputs[0]);
      const inChannels = inputDims[inputDims.length - 1];
      const outChannels = outputDims[outputDims.length - 1];
      const groups = this.getInt(node, 'group', 1);
      return [outChannels, kernelH, kernelW, inChannels / groups];
    } else {
      throw new Error(`Kernel shape cannot be inferred on a ${node.operator}`);
    }
  }
  // End of helper functions for openvino.Node

  // Helper functions for openvino.Tensor
  _getConstructorFromType(type) {
    switch (type) {
      case 'float32':
        return Float32Array;
      case 'I32':
        return Int32Array;
      default:
        throw new Error(`Tensor type ${type} is not supported`);
    }
  }

  /**
   * model: OpenVINOModel
   * arg: openvino.Argument|openvino.Node
   * dimHints?: number[] (NHWC)
   * OpenVino doesn't contain shape info for initializers. It needs to be
   * inferred from the given operators. `dimHints` is used for NCHW to NHWC
   * reordering. No reordering will be performed if the value is undefined.
   */
  getTensorInitializer(arg, dimHints) {
    const tensor = arg.connections[0].initializer;
    return this._getTensorData(tensor, dimHints);
  }

  /**
   * tensor: openvino.Node
   * dimHints?: number[] (NHWC)
   */
  getNodeInitilizer(node, dimHints) {
    // suppose each node has only one initializer
    const tensor = node._initializers[0].connections[0].initializer;
    return this._getTensorData(tensor, dimHints);
  }

  /**
   * tensor: openvino.Tensor
   * dimHints?: number[] (NHWC)
   */
  _getTensorData(tensor, dimHints) {
    // offset and size are encoded in `tensor.reference: string`
    const matches = tensor.reference.match(/offset: (\d+), size: (\d+)/);
    const [offset, size] = matches.slice(1, 3).map((x) => parseInt(x));
    const ctor = this._getConstructorFromType(tensor.type.dataType);
    const length = size / ctor.BYTES_PER_ELEMENT;
    const data = new ctor(this._weights, offset, length);
    if (typeof dimHints === 'undefined' || dimHints.length !== 4) {
      return data;
    }

    if (OpenVINOUtils.product(dimHints) !== length) {
      throw new Error(`Product of ${dimHints} doesn't match the length ${length}`);
    }
    // NCHW -> NHWC
    const nhwcData = new ctor(data.length);
    const [N, H, W, C] = dimHints;
    for (let n = 0; n < N; ++n) {
      for (let c = 0; c < C; ++c) {
        for (let h = 0; h < H; ++h) {
          for (let w = 0; w < W; ++w) {
            nhwcData[n*H*W*C + h*W*C + w*C + c] = data[n*C*H*W + c*H*W + h*W + w];
          }
        }
      }
    }
    return nhwcData;
  }

  getTensorGraphId(arg) {
    // graphId is unique in a graph and in the of form "layerId:portId"
    return arg.connections[0].id;
  }

  getTensorDataType(arg) {
    return this._getTensorType(arg).dataType;
  }

  getTensorShape(arg) {
    const dims = this._getTensorType(arg).shape.dimensions;
    if (dims.length !== 4) {
      return dims;
    } else {
      const [N, C, H, W] = dims;
      return [N, H, W, C];
    }
  }

  _getTensorType(arg) {
    return arg.connections[0].type;
  }
  // End of helper functions for openvino.Tensor
}


class OpenVINOUtils {
  static async loadOpenVINOModel() {
    let networkFile = '';
    let weightsFile = '';
    const argc = Array.from(arguments).length;
    if (argc === 1) {
      networkFile = arguments[0] + '.xml';
      weightsFile = arguments[0] + '.bin';
    } else if (argc === 2) {
      networkFile = arguments[0];
      weightsFile = arguments[1];
    } else {
      throw new Error('Invalid arguments.');
    }
    const networkText = await fetch(networkFile).then((res) => res.text());
    const weightsBuffer = await fetch(weightsFile).then((res) => res.arrayBuffer());
    return new OpenVINOModel(networkText, weightsBuffer);
  }

  static product(arr) {
    return arr.reduce((x, y) => x * y);
  }
}
