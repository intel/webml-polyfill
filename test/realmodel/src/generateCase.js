require('../lib/jsonOperation.js');
const fs = require('fs');
const path = require('path');
let filePath = path.join(__dirname, '..', 'model', JSON_DATA.getModelName(), `${JSON_DATA.getModelName()}.json`);
if (!fs.existsSync(filePath)) throw (`Can't get ${filePath}`);
let stream = fs.createReadStream(filePath, { flags: 'r', encoding: 'utf-8' });
let buf = '';

stream.on('data', function (d) {
  buf += d.toString();
});
stream.on('end', () => {
  buf = JSON.parse(buf);
  generateCase(buf);
});

function mkdirsSync(dirname) {
  if (fs.existsSync(dirname)) {
    return true;
  } else {
    if (mkdirsSync(path.dirname(dirname))) {
      fs.mkdirSync(dirname);
      return true;
    }
  }
}

async function saveToLocalFile(input) {
  let output = input.toString();
  let dataString;
  if (buf.operation.hasOwnProperty(input)) {
    dataString = buf.operation[input];
  } else if (buf.operands.hasOwnProperty(input)) {
    dataString = buf.operands[input];
  } else {
    throw ('please check input data');
  }

  let dataArray = [];
  for (let key in dataString) {
    dataArray.push(dataString[key]);
  }
  let saveFileDirs = path.join(__dirname, '..', 'testcase', 'res', `${JSON_DATA.getModelName()}`);
  mkdirsSync(saveFileDirs);
  let saveStream = fs.createWriteStream(path.join(saveFileDirs, output), { flags: 'w', encoding: 'utf-8' });
  saveStream.on('error', (err) => {
    console.error(err);
  });
  if (typeof (dataArray) === 'object') {
    saveStream.write(JSON.stringify(dataArray));
  } else {
    saveStream.write(dataArray);
  }
  saveStream.end();
}

async function saveCaseToLocal(input, output) {
  let saveFileDirs = path.join(__dirname, '..', 'testcase');
  mkdirsSync(saveFileDirs);
  let saveStream = fs.createWriteStream(path.join(saveFileDirs, output), { flags: 'w', encoding: 'utf-8' });
  saveStream.on('error', (err) => {
    console.error(err);
  });
  if (typeof (input) === 'object') {
    saveStream.write(JSON.stringify(input));
  } else {
    saveStream.write(input);
  }
  saveStream.end();
}

async function gettensorTypes(ids) {
  if (buf.tensorTypes.hasOwnProperty(ids)) {
    return buf.tensorTypes[ids].dimensions;
  }
}

async function getOperands(ids) {
  if (buf.operands.hasOwnProperty(ids)) {
    return buf.operands[ids];
  }
}

async function splitContext(context) {
  // context --> input.operations[i];
  let inputFile = context[1][0];
  let inputDims = (await gettensorTypes(inputFile));
  let outputFile = context[2][0];
  let outputDims = (await gettensorTypes(outputFile));
  await saveToLocalFile(inputFile);
  await saveToLocalFile(outputFile);
  switch(context[0]) {
  case 1: {
    let padding = (await getOperands(context[1][1]))[0];
    let stride = (await getOperands(context[1][5]))[0];
    let filter = (await getOperands(context[1][7]))[0];
    let activation = (await getOperands(context[1][9]))[0];
    let caseSample = `describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('Check result for AVERAGE_POOL_2D', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${outputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${inputDims}]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${outputDims}]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([${stride}]));
    model.setOperandValue(filter, new Int32Array([${filter}]));
    model.setOperandValue(padding, new Int32Array([${padding}]));
    model.setOperandValue(activation, new Int32Array([${activation}]));
    model.addOperation(nn.AVERAGE_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});`;
    await saveCaseToLocal(caseSample, `${JSON_DATA.getModelName()}-averagepool-${inputFile}.js`);
  } break;
  case 17:{
    let padding = (await getOperands(context[1][1]))[0];
    let stride = (await getOperands(context[1][5]))[0];
    let filter = (await getOperands(context[1][7]))[0];
    let activation = (await getOperands(context[1][9]))[0];
    let caseSample = `describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('Check result for max_pool_2d', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let i0_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      i0_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${outputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${inputDims}]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.INT32};
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${outputDims}]};
    let type2_length = product(type2.dimensions);
    let i0 = operandIndex++;
    model.addOperand(type0);
    let stride = operandIndex++;
    model.addOperand(type1);
    let filter = operandIndex++;
    model.addOperand(type1);
    let padding = operandIndex++;
    model.addOperand(type1);
    let activation = operandIndex++;
    model.addOperand(type1);
    let output = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(stride, new Int32Array([${stride}]));
    model.setOperandValue(filter, new Int32Array([${filter}]));
    model.setOperandValue(padding, new Int32Array([${padding}]));
    model.setOperandValue(activation, new Int32Array([${activation}]));
    model.addOperation(nn.MAX_POOL_2D, [i0, padding, padding, padding, padding, stride, stride, filter, filter, activation], [output]);
    model.identifyInputsAndOutputs([i0], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let i0_input = new Float32Array(i0_value);
    execution.setInput(0, i0_input);
    let output_output = new Float32Array(type2_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});`;
    await saveCaseToLocal(caseSample, `${JSON_DATA.getModelName()}-maxpool-${inputFile}.js`);
  } break;
  case 2: {
    let inputFile1 = context[1][1];
    let input1Dims = (await gettensorTypes(inputFile1));
    let axis = (await getOperands(context[1][2]))[0];
    await saveToLocalFile(inputFile1);
    let caseSample = `describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('Check result for concatenation', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let input1_value;
    let input2_value;
    let output_expect;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        file_data[j] = parseFloat(text[j]);
      }
      input1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile1}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      input2_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${outputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      output_expect = file_data;
    });
    let type2 = {type: nn.INT32};
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [${inputDims}]};
    let type1_length = product(type1.dimensions);
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${input1Dims}]};
    let type0_length = product(type0.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [${outputDims}]};
    let type3_length = product(type3.dimensions);
    let input1 = operandIndex++;
    model.addOperand(type0);
    let input2 = operandIndex++;
    model.addOperand(type1);
    let axis0 = operandIndex++;
    model.addOperand(type2);
    let output = operandIndex++;
    model.addOperand(type3);
    let input2_input = new Float32Array(input2_value);
    model.setOperandValue(input2, input2_input);
    model.setOperandValue(axis0, new Int32Array([${axis}]));
    model.addOperation(nn.CONCATENATION, [input1, input2, axis0], [output]);
    model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let input1_input = new Float32Array(input1_value);
    execution.setInput(0, input1_input);
    let output_output = new Float32Array(type3_length);
    execution.setOutput(0, output_output);
    await execution.startCompute();
    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output_output[i], output_expect[i]));
    }
  });
});`;
    await saveCaseToLocal(caseSample, `${JSON_DATA.getModelName()}-concatenation-${inputFile}.js`);
  } break;
  case 3: {
    let weightFile = context[1][1];
    let biasFile = context[1][2];
    let weight = (await gettensorTypes(weightFile));
    let bias = (await gettensorTypes(biasFile));
    let pad = (await getOperands(context[1][3]))[0];
    let stride = (await getOperands(context[1][7]))[0];
    let act = (await getOperands(context[1][9]))[0];
    await saveToLocalFile(weightFile);
    await saveToLocalFile(biasFile);
    let caseSample = `describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('Check result for conv_2d', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op4_expect;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${outputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op4_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${inputDims}]};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [${outputDims}]};
    let type1_length = product(type1.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${bias}]};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.INT32};
    let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [${weight}]};
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type4);
    let op3 = operandIndex++;
    model.addOperand(type2);
    let pad0 = operandIndex++;
    model.addOperand(type3);
    let act = operandIndex++;
    model.addOperand(type3);
    let stride = operandIndex++;
    model.addOperand(type3);
    let op4 = operandIndex++;
    model.addOperand(type1);
    let op2value;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${weightFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op2value = file_data;
    });
    let op3value;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${biasFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3value = file_data;
    });
    model.setOperandValue(op2, new Float32Array(op2value));
    model.setOperandValue(op3, new Float32Array(op3value));
    model.setOperandValue(pad0, new Int32Array([${pad}]));
    model.setOperandValue(act, new Int32Array([${act}]));
    model.setOperandValue(stride, new Int32Array([${stride}]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, pad0, pad0, pad0, pad0, stride, stride, act], [op4]);
    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Float32Array(type1_length);
    execution.setOutput(0, op4_output);
    await execution.startCompute();
    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});`;
    await saveCaseToLocal(caseSample, `${JSON_DATA.getModelName()}-conv2d-${weightFile}.js`);
  } break;
  case 22: {
    let shapeDic = await getOperands(context[1][1]);
    let shapeLen = Object.keys(shapeDic).length;
    let shapeValues = Object.values(shapeDic);
    let caseSample = `describe('CTS Real Model Test', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();
  it('Check result for reshape', async function() {
    this.timeout(120000);
    let model = await nn.createModel(options);
    let operandIndex = 0;
    let op1_value;
    let op3_expect;
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${inputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op1_value = file_data;
    });
    await fetch('./realmodel/testcase/res/${JSON_DATA.getModelName()}/${outputFile}').then((res) => {
      return res.json();
    }).then((text) => {
      let file_data = new Float32Array(text.length);
      for (let j in text) {
        let b = parseFloat(text[j]);
        file_data[j] = b;
      }
      op3_expect = file_data;
    });
    let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${inputDims}]};
    let type0_length = product(type0.dimensions);
    let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${outputDims}]};
    let type2_length = product(type2.dimensions);
    let type1 = {type: nn.TENSOR_INT32, dimensions: [${shapeLen}]};
    let type1_length = product(type1.dimensions);
    let op1 = operandIndex++;
    model.addOperand(type0);
    let op2 = operandIndex++;
    model.addOperand(type1);
    let op3 = operandIndex++;
    model.addOperand(type2);
    model.setOperandValue(op2, new Int32Array([${shapeValues}]));
    model.addOperation(nn.RESHAPE, [op1, op2], [op3]);
    model.identifyInputsAndOutputs([op1], [op3]);
    await model.finish();
    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();
    let execution = await compilation.createExecution();
    let op1_input = new Float32Array(op1_value);
    execution.setInput(0, op1_input);
    let op3_output = new Float32Array(type2_length);
    execution.setOutput(0, op3_output);
    await execution.startCompute();
    for (let i = 0; i < type2_length; ++i) {
      assert.isTrue(almostEqualCTS(op3_output[i], op3_expect[i]));
    }
  });
});`;
    await saveCaseToLocal(caseSample, `${JSON_DATA.getModelName()}-reshape-${inputFile}.js`);
  }break;
  }
}

async function findSync(startPath) {
  let result=[];
  String.prototype.endWith = function (endStr) {
    let d = this.length - endStr.length;
    return (d >= 0 && this.lastIndexOf(endStr) == d);
  };
  let files = fs.readdirSync(startPath);
  files.forEach((val,index) => {
    if ((val.indexOf(JSON_DATA.getModelName()) == 0 ) && val.endsWith('.js')) {
      result.push(val);
    }
  });
  let saveFileDirs = path.join(__dirname, '..', 'testcase');
  let saveStream = fs.createWriteStream(
    path.join(saveFileDirs, `${JSON_DATA.getModelName()}.txt`),
    { flags: 'w', encoding: 'utf-8' }
  );
  saveStream.on('error', (err) => {
    console.error(err);
  });
  saveStream.write(JSON.stringify(result));
  saveStream.end();
}

async function generateCase(input) {
  if (input.hasOwnProperty('operations')) {
    for (let i = 0; i < input.operations.length; i++) {
      await splitContext(input.operations[i]);
    }
    await findSync(path.join(__dirname, '..', 'testcase'));
  }
}
