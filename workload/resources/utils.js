class Logger {
  constructor($dom) {
    this.$dom = $dom;
    this.indent = 0;
  }

  log(message) {
    console.log(message);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + message}`;
  }

  error(err) {
    console.error(err);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + err.message}`;
  }

  group(name) {
    console.group(name);
    this.log('');
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + name}`;
    this.indent++;
  }

  groupEnd() {
    console.groupEnd();
    this.indent--;
  }
}

class Benchmark {
  constructor(modelName, backend, iterations) {
    this.modelName = modelName;
    this.backend = backend;
    this.iterations = iterations;
  }

  async runAsync() {
    await this.setupAsync();
    let results = await this.executeAsync();
    return this.handleResults(results);
  }

  /**
   * Setup model
   * @returns {Promise<void>}
   */
  async setupAsync() {
    throw Error("Not Implemented");
  }

  onExecuteSingle(iteration) {}

  /**
   * Execute model
   * @returns {Promise<void>}
   */
  async executeSingleAsync() {
    throw Error('Not Implemented');
  }

  async executeAsync() {
    let results = [];
    for (let i = 0; i < this.iterations; i++) {
      this.onExecuteSingle(i);
      let tStart = performance.now();
      await this.executeSingleAsync();
      let elapsedTime = performance.now() - tStart;
      results.push(elapsedTime);
    }
    return results;
  }

  handleResults(results) {
    throw Error('Not Implemented');
  }

  /**
   * Finalize
   * @returns {Promise<void>}
   */
  finalize() {}
}

function setInputTensor(pixels, imageChannels, height, width, channels, channelScheme, mean, std, inputTensor) {
  if (channelScheme === 'RGB') {
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y * width * imageChannels + x * imageChannels + c];
          inputTensor[y * width * channels + x * channels + c] = (value - mean[c]) / std[c];
        }
      }
    }
  } else if (channelScheme === 'BGR') {
    // NHWC layout
    for (let y = 0; y < height; ++y) {
      for (let x = 0; x < width; ++x) {
        for (let c = 0; c < channels; ++c) {
          let value = pixels[y * width * imageChannels + x * imageChannels + (channels - c - 1)];
          inputTensor[y * width * channels + x * channels + c] = (value - mean[c]) / std[c];
        }
      }
    }
  }
}

async function loadImage(url) {
  let image = new Image();
  let promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
      resolve(image);
    };
  });
  image.src = url;
  return promise;
}

async function loadUrl(url, binary) {
  return new Promise((resolve, reject) => {
    let request = new XMLHttpRequest();
    request.open('GET', url, true);
    if (binary) {
      request.responseType = 'arraybuffer';
    }
    request.onload = (ev) => {
      if (request.readyState === 4) {
        if (request.status === 200) {
          resolve(request.response);
        } else {
          reject(new Error(`Failed to load ${url} status: ${request.status}`));
        }
      }
    };
    request.send();
  });
}

async function loadModelAndLabels(model, label=null) {
  let arrayBuffer, bytes, text;
  let url = '../examples/util/';
  if (model.toLowerCase().startsWith("https://") || model.toLowerCase().startsWith("http://")) {
    arrayBuffer = await this.loadUrl(model, true);
  } else {
    arrayBuffer = await this.loadUrl(url + model, true);
  }
  bytes = new Uint8Array(arrayBuffer);
  if (label != null && (label.toLowerCase().startsWith("https://") || label.toLowerCase().startsWith("http://"))) {
    text = await this.loadUrl(label);
  } else {
    text = label ? await this.loadUrl(url + label) : null;
  }

  return {
    bytes: bytes,
    text: text
  };
}

function summarize(results) {
  if (results.length !== 0) {
    // remove first run, which is regarded as "warming up" execution
    results.shift();
    let d = results.reduce((d, v) => {
      d.sum += v;
      d.sum2 += v * v;
      return d;
    }, {
      sum: 0,
      sum2: 0
    });
    let mean = d.sum / results.length;
    let std = Math.sqrt((d.sum2 - results.length * mean * mean) / (results.length - 1));
    return {
      mean: mean,
      std: std
    };
  } else {
    return null;
  }
}

function summarizeProf(results) {
  const lines = [];
  if (!results) {
    return lines;
  }
  lines.push(`Execution calls: ${results.epochs} (omitted ${results.warmUpRuns} warm-up runs)`);
  lines.push(`Supported Ops: ${results.supportedOps.join(', ') || 'None'}`);
  lines.push(`Mode: ${results.mode}`);

  let polyfillTime = 0;
  let webnnTime = 0;
  for (const t of results.timings) {
    lines.push(`${t.elpased.toFixed(8).slice(0, 7)} ms\t- (${t.backend}) ${t.summary}`);
    if (t.backend === 'WebNN') {
      webnnTime += t.elpased;
    } else {
      polyfillTime += t.elpased;
    }
  }
  lines.push(`Polyfill time: ${polyfillTime.toFixed(5)} ms`);
  lines.push(`WebNN time: ${webnnTime.toFixed(5)} ms`);
  lines.push(`Sum: ${(polyfillTime + webnnTime).toFixed(5)} ms`);
  return lines;
}

function getModelInfoDict(modelsArray, modelName) {
  return modelsArray.filter(f => f.modelName == modelName)[0];
}
