var fs = require('fs');
let filePath = process.argv[2];
var stream = fs.createReadStream(filePath, { flags: 'r', encoding: 'utf-8' });
var buf = '';

stream.on('data', function (d) {
  buf += d.toString();
});
stream.on('end', () => {
  buf = JSON.parse(buf);
  generateCase(buf);
  downloadSource(buf);
});
async function saveToLocalFile(input, output) {
  var saveStream = fs.createWriteStream(output, { flags: 'w', encoding: 'utf-8' });
  saveStream.on('error', (err) => {
    console.log(err);
  });
  if (typeof (input) === 'object') {
    saveStream.write(JSON.stringify(input));
  } else {
    saveStream.write(input);
  }
  saveStream.end();
}
async function downloadSource(input) {
  let dataArray = [];
  let Source = input['Source'];
  for (let sourceName in Source){
    for (let key in Source[sourceName]){
      if (Source[sourceName].hasOwnProperty(key)) {
        dataArray.push(Source[sourceName][key]);
      }
    }
    await saveToLocalFile(dataArray, `../../testcase/res/${sourceName}`);
  }
}
async function generateCase(input) {
  if (input.hasOwnProperty('Case')) {
    let Case = input['Case'];
    let Source = input['Source'];
    for (let opType in Case) {
      switch (opType) {
      case 'Conv': {
        //node, bias, weight, options
        // options: inputDims, outputDims, Dims, pads, strides, relu, fuserelu
        for (let caseName in Case[opType]) {
          let caseSample = `
                        describe('CTS Real Model Test', function() {
                            const assert = chai.assert;
                            const nn = navigator.ml.getNeuralNetworkContext();

                            it('Check result for conv_2d by ${caseName}', async function() {
                              this.timeout(120000);
                              let model = await nn.createModel(options);
                              let operandIndex = 0;

                              let op1_value;
                              let op4_expect;

                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][0]}').then((res) => {
                                return res.json();
                              }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                  let b = parseFloat(text[j]);
                                  file_data[j] = b;
                                }
                                op1_value = file_data;
                              });

                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['options'][6]}').then((res) => {
                                return res.json();
                              }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                  let b = parseFloat(text[j]);
                                  file_data[j] = b;
                                }
                                op4_expect = file_data;
                              });

                              let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][0]}]};
                              let type0_length = product(type0.dimensions);
                              let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][1]}]};
                              let type1_length = product(type1.dimensions);
                              let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][2][0]}]};
                              let type2_length = product(type2.dimensions);
                              let type3 = {type: nn.INT32};
                              let type4 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][2]}]};

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
                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][1]}').then((res) => {
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
                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][2]}').then((res) => {
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
                              model.setOperandValue(pad0, new Int32Array([${Case[opType][caseName]['options'][3][0]}]));
                              model.setOperandValue(act, new Int32Array([${Case[opType][caseName]['options'][5]}]));
                              model.setOperandValue(stride, new Int32Array([${Case[opType][caseName]['options'][4][0]}]));
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
                          });
                        `;
          await saveToLocalFile(caseSample, `../../testcase/${caseName}.js`);
        }
      } break;
      case 'Concat': {
        // options: input1Dims, input2Dims, outputDims, axis
        for (let caseName in Case[opType]) {
          let caseSample = `
                        describe('CTS Real Model Test', function() {
                        const assert = chai.assert;
                        const nn = navigator.ml.getNeuralNetworkContext();

                            it('Check result for concatenation by ${caseName}', async function() {
                            this.timeout(120000);
                            let model = await nn.createModel(options);
                            let operandIndex = 0;

                            let input1_value;
                            let input2_value;
                            let output_expect;

                            await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][0]}').then((res) => {
                            return res.json();
                            }).then((text) => {
                            let file_data = new Float32Array(text.length);
                            for (let j in text) {
                                file_data[j] = parseFloat(text[j]);
                            }
                            input1_value = file_data;
                            });
                            await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][1]}').then((res) => {
                            return res.json();
                            }).then((text) => {
                            let file_data = new Float32Array(text.length);
                            for (let j in text) {
                                let b = parseFloat(text[j]);
                                file_data[j] = b;
                            }
                            input2_value = file_data;
                            });
                            await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['output'][0]}').then((res) => {
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
                            let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][0]}]};
                            let type1_length = product(type1.dimensions);
                            let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][1]}]};
                            let type0_length = product(type0.dimensions);
                            let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][2]}]};
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

                            model.setOperandValue(axis0, new Int32Array([${Case[opType][caseName]['options'][3]}]));
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
                        });
                        `;
          await saveToLocalFile(caseSample, `../../testcase/${caseName}.js`);
        }
      } break;
      case 'Reshape': {
        // options: inputDims, outputDims
        for (let caseName in Case[opType]) {
          let caseSample = `
                        describe('CTS Real Model Test', function() {
                            const assert = chai.assert;
                            const nn = navigator.ml.getNeuralNetworkContext();

                            it('Check result for reshape by ${caseName}', async function() {
                                this.timeout(120000);
                                let model = await nn.createModel(options);
                                let operandIndex = 0;

                                let op1_value;
                                let op3_expect;
                                await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][0]}').then((res) => {
                                return res.json();
                                }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                    let b = parseFloat(text[j]);
                                    file_data[j] = b;
                                }
                                op1_value = file_data;
                                });
                                await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['output'][0]}').then((res) => {
                                return res.json();
                                }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                    let b = parseFloat(text[j]);
                                    file_data[j] = b;
                                }
                                op3_expect = file_data;
                                });

                                let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][0]}]};
                                let type0_length = product(type0.dimensions);
                                let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][1]}]};
                                let type2_length = product(type2.dimensions);
                                let type1 = {type: nn.TENSOR_INT32, dimensions: [${Case[opType][caseName]['options'][0][0]}]};
                                let type1_length = product(type1.dimensions);

                                let op1 = operandIndex++;
                                model.addOperand(type0);
                                let op2 = operandIndex++;
                                model.addOperand(type1);
                                let op3 = operandIndex++;
                                model.addOperand(type2);

                                model.setOperandValue(op2, new Int32Array([1000]));
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
                            });
                        `;
          await saveToLocalFile(caseSample, `../../testcase/${caseName}.js`);
        }
      } break;
      case 'AveragePool':
      case 'MaxPool': {
        //options: inputDims, outputDims, pads, strides, kernelshape
        for (let caseName in Case[opType]) {
          let caseSample = `
                        describe('CTS Real Model Test', function() {
                            const assert = chai.assert;
                            const nn = navigator.ml.getNeuralNetworkContext();

                            it('Check result for max_pool_2d by ${caseName}', async function() {
                              this.timeout(120000);
                              let model = await nn.createModel(options);
                              let operandIndex = 0;

                              let i0_value;
                              let output_expect;

                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['input'][0]}').then((res) => {
                                return res.json();
                              }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                  let b = parseFloat(text[j]);
                                  file_data[j] = b;
                                }
                                i0_value = file_data;
                              });

                              await fetch('./realmodel/testcase/res/${Case[opType][caseName]['node']['output'][0]}').then((res) => {
                                return res.json();
                              }).then((text) => {
                                let file_data = new Float32Array(text.length);
                                for (let j in text) {
                                  let b = parseFloat(text[j]);
                                  file_data[j] = b;
                                }
                                output_expect = file_data;
                              });

                              let type0 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][0]}]};
                              let type0_length = product(type0.dimensions);
                              let type1 = {type: nn.INT32};
                              let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${Case[opType][caseName]['options'][1]}]};
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

                              model.setOperandValue(stride, new Int32Array([${Case[opType][caseName]['options'][3][0]}]));
                              model.setOperandValue(filter, new Int32Array([${Case[opType][caseName]['options'][4][0]}]));
                              model.setOperandValue(padding, new Int32Array([${Case[opType][caseName]['options'][2][0]}]));
                              model.setOperandValue(activation, new Int32Array([${Case[opType][caseName]['options'][2][0]}]));
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
                          });
                        `;
          await saveToLocalFile(caseSample, `../../testcase/${caseName}.js`);
        }
      } break;
      }
    }
  }
}
