class SpeechCommandsRunner extends BaseRunner {
  constructor() {
    super();
    this._labels = null;
  }

  _setLabels = (labels) => {
    this._labels = labels;
  };

  _getLabels = async (url) => {
    const result = await this._loadURL(url);
    this._setLabels(result.split('\n'));
    console.log(`labels: ${this._labels}`);
  };

  _getOtherResources = async () => {
    await this._getLabels(this._currentModelInfo.labelsFile);
  };

  _updateOutput = (output) => {
    output.labels = this._labels;
  };
}