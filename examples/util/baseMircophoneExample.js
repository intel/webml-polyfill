class baseMircophoneExample extends baseExample {
  constructor(models) {
    super(models);
  }

  _getMediaConstraints = () => {
    const constraints = {audio: true};
    return constraints;
  };
};