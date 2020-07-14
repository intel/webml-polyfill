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
  _getMediaStream = async () => {
    let stream = await navigator.mediaDevices.getUserMedia({audio: true});
    return stream;
  };
};