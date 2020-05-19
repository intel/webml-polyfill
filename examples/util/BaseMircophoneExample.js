class BaseMircophoneExample extends BaseExample {
  constructor(models) {
    super(models);
  }

  /** @override */
  _getDefaultInputType = () => {
    return 'audio';
  };

  /** @override */
  _getDefaultInputMediaType = () => {
    return 'microphone';
  };

  /** @override */
  _getMediaConstraints = () => {
    const constraints = {audio: true};
    return constraints;
  };
};