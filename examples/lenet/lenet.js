
var nn = navigator.ml.getNeuralNetworkContext('v2');

class Lenet {
  constructor(url) {
    this.url_ = url;
    this.model_ = null;
    this.compilation_ = null;
  }

  async load() {
    const response = await fetch(this.url_);
    const arrayBuffer = await response.arrayBuffer();

    // this.model_ = await nn.createModel([{name: 'output', operand: output}]);
  }

  async compile(options) {
    // this.compilation_ = await this.model_.createCompilation(options);
  }

  async predict(input) {
    // const execution = await this.compilation_.createExecution();

    // await execution.startCompute();

    // return result;
  }
}
