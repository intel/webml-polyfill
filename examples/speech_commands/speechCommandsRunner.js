class speechCommandsRunner extends baseRunner {
  constructor() {
    super();
    this._labels = null;
  }

  _setLabels = (labels) => {
    this._labels = labels;
  };

  _getLabelsAsync = async (url) => {
    const result = await this._loadURLAsync(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  _getOtherResourcesAsync = async () => {
    await this._getLabelsAsync(this._currentModelInfo.labelsFile);
  };

  _updateOutput = (output) => {
    output.labels = this._labels;
  };
}