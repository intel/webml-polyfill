class Utils {
  constructor(canvas) {
    this.rawModel;
    this.model;
    this.inputTensor = [];
    this.outputTensor = [];
    this.inputSize;
    this.outputSize;
    this.boxH;
    this.boxW;
    this.modelID;
    this.postOptions;
    this.preOptions;
    this.canvasElement = canvas;
    this.canvasContext = this.canvasElement.getContext('2d');
    this.updateProgress;
    this.loaded = false;
    this.initialized = false;
    this.resolveGetRequiredOps = null;
    this.outstandingRequest = null;
  }

  async loadModel(newModel) {
    if (this.loaded && this.modelFile === newModel.modelFile) {
      return 'LOADED';
    }
    // reset all states
    this.loaded = this.initialized = false;
    this.backend = this.prefer = '';

    // set new model params
    this.inputSize = newModel.inputSize;
    this.outputSize = newModel.outputSize;
    this.modelID = newModel.modelId;
    this.modelFile = newModel.modelFile;
    this.postOptions = newModel.postOptions || {};
    this.preOptions = newModel.preOptions || {};
    this.inputTensor = [new Float32Array(newModel.inputSize.reduce((x,y) => x*y))];
    this.outputTensor = [new Float32Array(newModel.outputSize.reduce((x,y) => x*y))];

    this.canvasElement.width = newModel.inputSize[2];
    this.boxW = newModel.inputSize[2];
    this.canvasElement.height = newModel.inputSize[1];
    this.boxH = newModel.inputSize[1];

    let arrayBuffer = await this.loadUrl(this.modelFile, true, true);
    let weightsBuffer = new Uint8Array(arrayBuffer);
    let networkFile = this.modelFile.replace(/bin$/, 'xml');
    let networkText = await this.loadUrl(networkFile, false, false);
    this.rawModel = new OpenVINOModel(networkText, weightsBuffer.buffer);
    this.rawModel._rawFormat = 'OPENVINO';

    this.loaded = true;
    return 'SUCCESS';
  }

  async init(backend, prefer) {
    if (!this.loaded) {
      return 'NOT_LOADED';
    }
    if (this.initialized && backend === this.backend && prefer === this.prefer) {
      return 'INITIALIZED';
    }
    this.initialized = false;
    this.backend = backend;
    this.prefer = prefer;
    let configs = {
      rawModel: this.rawModel,
      backend: backend,
      prefer: prefer,
    };
    this.model = new OpenVINOModelImporter(configs);
    await this.model.createCompiledModel();
    let start = performance.now();
    let result = await this.model.compute(this.inputTensor, this.outputTensor);
    let elapsed = performance.now() - start;
    console.log(`warmup time: ${elapsed.toFixed(2)} ms`);
    this.initialized = true;

    if (this.resolveGetRequiredOps) {
      this.resolveGetRequiredOps(this.model.getRequiredOps());
    }

    return 'SUCCESS';
  }

  async getRequiredOps() {
    if (!this.initialized) {
      return new Promise(resolve => this.resolveGetRequiredOps = resolve);
    } else {
      return this.model.getRequiredOps();
    }
  }

  getSubgraphsSummary() {
    if (this.model._backend !== 'WebML' &&
        this.model &&
        this.model._compilation &&
        this.model._compilation._preparedModel) {
      return this.model._compilation._preparedModel.getSubgraphsSummary();
    } else {
      return [];
    }
  }

  euclideanDistance(embeddings1, embeddings2) {
    let embeddingSum = 0;
    for (let i = 0; i < embeddings1.length; i++) {
      embeddingSum = embeddingSum + Math.pow((embeddings1[i] - embeddings2[i]), 2);
    }

    return Math.sqrt(embeddingSum);
  }

  cosineDistance(embeddings1, embeddings2) {
    let dotSum = 0;
    let norm1 = 0;
    let norm2 = 0;
    for (let i = 0; i < embeddings1.length; i++) {
      dotSum = dotSum + embeddings1[i] * embeddings2[i];
      norm1 = norm1 + Math.pow(embeddings1[i], 2);
      norm2 = norm2 + Math.pow(embeddings2[i], 2);
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    let cosine = dotSum / (norm1 * norm2);

    return 1 - cosine;
  }

  // Embeddings L2-Normalization
  L2Normalization(embeddings) {
    // norm(L2) = (|x0|^2 + |x1|^2 + |x2|^2 + |xi|^2)^1/2
    let embeddingSum = 0;
    for (let i = 0; i < embeddings.length; i++) {
      if (embeddings[i] !== 0) {
        embeddingSum = embeddingSum + Math.pow(Math.abs(embeddings[i]), 2);
      }
    }
    let L2 = Math.sqrt(embeddingSum);

    let embeddingsNorm = new Float32Array(embeddings.length);
    for (let i = 0; i < embeddings.length; i++) {
      if (embeddings[i] !== 0) {
        embeddingsNorm[i] = (embeddings[i] / L2).toFixed(10);
      } else {
        embeddingsNorm[i] = 0;
      }
    }

    return embeddingsNorm;
  }

  getBoxs(boxs) {
    let boxMean = (this.boxH + this.boxW) / 2;
    let boxsTmp = new Array();
    for (let box of boxs) {
      let h = box[3] - box[2];
      let w = box[1] - box[0];
      let mean = (h + w) / 2;

      let sideW = Math.floor(this.boxW * mean / boxMean / 2);
      let sideH = Math.floor(this.boxH * mean / boxMean / 2);

      let centerW = Math.floor((box[1] + box[0]) / 2);
      let centerH = Math.floor((box[3] + box[2]) / 2);
      let boxTmp = new Array();
      boxTmp[0] = centerW - sideW;
      boxTmp[1] = centerW + sideW;
      boxTmp[2] = centerH - sideH;
      boxTmp[3] = centerH + sideH;
      boxsTmp.push(boxTmp);
    }

    return boxsTmp;
  }

  getClass(targetEmbeddings, searchEmbeddings) {
    let results = new Array();
    let distanceMap = new Map();

    for (let i in targetEmbeddings) {
      for (let j in searchEmbeddings) {
        // Set default status 'unknown' as 'X'
        results[j] = 'X';

        let distance;
        if (this.postOptions.distanceMetric === "euclidean") {
          let [...targetEmbeddingsTmp] = Float32Array.from(this.L2Normalization(targetEmbeddings[i]));
          let [...searchEmbeddingsTmp] = Float32Array.from(this.L2Normalization(searchEmbeddings[j]));
          distance = this.euclideanDistance(targetEmbeddingsTmp, searchEmbeddingsTmp);
        } else if (this.postOptions.distanceMetric === "cosine") {
          distance = this.cosineDistance(targetEmbeddings[i], searchEmbeddings[j]);
        } else {
          throw new Error(`Not support distance metric: ${this.postOptions.distanceMetric}`);
        }

        if (!distanceMap.has(j)) distanceMap.set(j, new Map());
        distanceMap.get(j).set(i, distance);
      }
    }

    console.dir(distanceMap);

    for (let key1 of distanceMap.keys()) {
      let num = null;
      let minDis = null;
      for (let [key2, value2] of distanceMap.get(key1).entries()) {
        if (minDis == null) {
          num = key2;
          minDis = value2;
        } else {
          if (minDis > value2) {
            num = key2;
            minDis = value2;
          }
        }
      }

      if (results[key1] === 'X' && minDis < this.postOptions.threshold) {
        results[key1] = parseInt(num) + 1;
      }
    }

    return results;
  }

  async predict(source, boxes) {
    if (!this.initialized) return;

    let start = performance.now();

    let embeddings = new Array();
    for (let box of boxes) {
      this.canvasContext.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
      this.canvasContext.drawImage(source, box[0], box[2],
                                   box[1]-box[0], box[3]-box[2], 0, 0,
                                   this.canvasElement.width,
                                   this.canvasElement.height);
      this.prepareInputTensor(this.inputTensor, this.canvasElement);
      await this.model.compute(this.inputTensor, this.outputTensor);
      let [...normEmbedding] = Float32Array.from(this.outputTensor[0]);
      embeddings.push(normEmbedding);
    }

    let elapsed = performance.now() - start;
    console.log(`Face Recognition Inference time: ${elapsed.toFixed(2)} ms`);

    return {
      embedding: embeddings,
      time: elapsed.toFixed(2)
    };
  }

  async loadRawModel(modelUrl) {
    let arrayBuffer = await this.loadUrl(modelUrl, true, true);
    let bytes = new Uint8Array(arrayBuffer);
    return {bytes: bytes};
  }

  async loadUrl(url, binary, progress) {
    return new Promise((resolve, reject) => {
      if (this.outstandingRequest) {
        this.outstandingRequest.abort();
      }
      let request = new XMLHttpRequest();
      this.outstandingRequest = request;
      request.open('GET', url, true);
      if (binary) {
        request.responseType = 'arraybuffer';
      }
      request.onload = function(ev) {
        this.outstandingRequest = null;
        if (request.readyState === 4) {
          if (request.status === 200) {
              resolve(request.response);
          } else {
              reject(new Error('Failed to load ' + url + ' status: ' + request.status));
          }
        }
      };
      if (progress && typeof this.updateProgress !== 'undefined') {
        request.onprogress = this.updateProgress;
      }
      request.send();
    });
  }

  prepareInputTensor(tensors, canvas) {
    let tensor = tensors[0];
    let W = this.inputSize[2];
    let H = this.inputSize[1];
    let C = this.inputSize[3];
    let imageC = 4; // NHWC -- RGBA
    let channelScheme = this.preOptions.channelScheme;
    let dataFormat = this.preOptions.format;
    let mean = this.preOptions.mean;
    let std = this.preOptions.std;

    if (canvas.width !== W || canvas.height !== H) {
      throw new Error(`canvas.width(${canvas.width}) is not ${W} or canvas.height(${canvas.height}) is not ${H}`);
    }

    let context = canvas.getContext('2d');
    let pixels = context.getImageData(0, 0, W, H).data;

    if (dataFormat === 'NHWC') {
      if (channelScheme === 'RGB') {
        for (let h = 0; h < H; ++h) {
          for (let w = 0; w < W; ++w) {
            for (let c = 0; c < C; ++c) {
              let value = pixels[h * W * imageC + w * imageC + c];
              tensor[h * W * C + w * C + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else if (channelScheme === 'BGR') {
        for (let h = 0; h < H; ++h) {
          for (let w = 0; w < W; ++w) {
            for (let c = 0; c < C; ++c) {
              let value = pixels[h * W * imageC + w * imageC + (C - c - 1)];
              tensor[h * W * C + w * C + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else {
        throw new Error(`Unknown color channel scheme ${channelScheme}`);
      }
    } else if (dataFormat === 'NCHW') {
      if (channelScheme === 'RGB') {
        for (let c = 0; c < C; ++c) {
          for (let h = 0; h < H; ++h) {
            for (let w = 0; w < W; ++w) {
              let value = pixels[h * W * imageC + w * imageC + c];
              tensor[h * W * C + w * C + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else if (channelScheme === 'BGR') {
        for (let c = 0; c < C; ++c) {
          for (let h = 0; h < H; ++h) {
            for (let w = 0; w < W; ++w) {
              let value = pixels[h * W * imageC + w * imageC + (C - c - 1)];
              tensor[h * W * C + w * C + c] = (value - mean[c]) / std[c];
            }
          }
        }
      } else {
        throw new Error(`Unknown color channel scheme ${channelScheme}`);
      }
    } else {
      throw new Error(`Unknown data format ${dataFormat}`);
    }
  }

  deleteAll() {
    if (this.model._backend != 'WebML') {
      this.model._compilation._preparedModel._deleteAll();
    }
  }
}
