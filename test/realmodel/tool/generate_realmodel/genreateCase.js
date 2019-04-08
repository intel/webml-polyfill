const fs = require('fs');
const path = require('path');
let filePath = path.join(__dirname, 'casePrototypeData.json')
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
   let output = input;
   let dataString;
   if (buf.operation.hasOwnProperty(input)) {
    dataString = buf.operation[input];
   } else if (buf.tensorId.hasOwnProperty(input)) {
    dataString = buf.operands[buf.tensorId[input]['id']];
   } else {
     throw error ('please check input data');
   }

   let dataArray = [];
   for (let key in dataString) {
     dataArray.push(dataString[key]);
   }
  let saveFileDirs = path.join(process.cwd(), 'testcase', 'res');
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
   let saveFileDirs = path.join(process.cwd(), 'testcase');
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

 async function getTensorString(ids) {
   let tensorIdString;
   Object.keys(buf.tensorId).forEach((k) => {
     if(buf.tensorId[k]['id'] == ids) {
       tensorIdString = [k, buf.tensorId[k]['type']];
     }
   })
   return tensorIdString;
 }

 async function getOperandsValue(data) {
   // data --> input.operations[i][1][3]
   return buf.operands[data];
 }

 async function splitContext(context) {
   // context --> input.operations[i];
   let inputName = (await getTensorString(context[1][0]))[0];
   let inputDims = (await getTensorString(context[1][0]))[1].dimensions;
   let outputName = (await getTensorString(context[2][0]))[0];
   let outputDims = (await getTensorString(context[2][0]))[1].dimensions;
   await saveToLocalFile(inputName);
   await saveToLocalFile(outputName);
   switch(context[0]) {
     case 1: 
     case 17:{
       let padding = (await getOperandsValue(context[1][1]))[0];
       let padding1 = (await getOperandsValue(context[1][2]))[0];
       let padding2 = (await getOperandsValue(context[1][3]))[0];
       let padding3 = (await getOperandsValue(context[1][4]))[0];
       let stride = (await getOperandsValue(context[1][5]))[0];
       let stride1 = (await getOperandsValue(context[1][6]))[0];
       let filter = (await getOperandsValue(context[1][7]))[0];
       let filter1 = (await getOperandsValue(context[1][8]))[0];
       let activation = (await getOperandsValue(context[1][9]))[0];
      let caseSample = `
      describe('CTS Real Model Test', function() {
       const assert = chai.assert;
       const nn = navigator.ml.getNeuralNetworkContext();
       it('Check result for max_pool_2d by ${context[0]}', async function() {
         this.timeout(120000);
         let model = await nn.createModel(options);
         let operandIndex = 0;
         let i0_value;
         let output_expect;
         await fetch('./realmodel/testcase/res/${inputName}').then((res) => {
           return res.json();
         }).then((text) => {
           let file_data = new Float32Array(text.length);
           for (let j in text) {
             let b = parseFloat(text[j]);
             file_data[j] = b;
           }
           i0_value = file_data;
         });
         await fetch('./realmodel/testcase/res/${outputName}').then((res) => {
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
     });
      `;
      await saveCaseToLocal(caseSample, `average-max${inputName}.js`)
     } break;
     case 2: {
      let input1Name = (await getTensorString(context[1][1]))[0];
      let input1Dims = (await getTensorString(context[1][1]))[1].dimensions;
      let axis = (await getOperandsValue(context[1][2]))[0];
      await saveToLocalFile(input1Name);
      let caseSample = `
         describe('CTS Real Model Test', function() {
          const assert = chai.assert;
          const nn = navigator.ml.getNeuralNetworkContext();
              it('Check result for concatenation by ${arguments[0][1]}', async function() {
              this.timeout(120000);
              let model = await nn.createModel(options);
              let operandIndex = 0;
              let input1_value;
              let input2_value;
              let output_expect;
              await fetch('./realmodel/testcase/res/${inputName}').then((res) => {
              return res.json();
              }).then((text) => {
              let file_data = new Float32Array(text.length);
              for (let j in text) {
                  file_data[j] = parseFloat(text[j]);
              }
              input1_value = file_data;
              });
              await fetch('./realmodel/testcase/res/${input1Name}').then((res) => {
              return res.json();
              }).then((text) => {
              let file_data = new Float32Array(text.length);
              for (let j in text) {
                  let b = parseFloat(text[j]);
                  file_data[j] = b;
              }
              input2_value = file_data;
              });
              await fetch('./realmodel/testcase/res/${outputName}').then((res) => {
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
          });
         `;
         await saveCaseToLocal(caseSample, `concat${inputName}.js`)
     } break;
     case 3: {
       let weightFile = (await getTensorString(context[1][1]))[0];            //  squeezenet0_conv0_weight
       let biasFile = (await getTensorString(context[1][2]))[0];              //  squeezenet0_conv0_bias
       let weight = (await getTensorString(context[1][1]))[1].dimensions;     //  [ 64, 3, 3, 3 ]
       let bias = (await getTensorString(context[1][2]))[1].dimensions;       //  [ 64 ]
       let pad = (await getOperandsValue(context[1][3]))[0];                  //  0
       let pad1 = (await getOperandsValue(context[1][4]))[0];
       let pad2 = (await getOperandsValue(context[1][5]))[0];
       let pad3 = (await getOperandsValue(context[1][6]))[0];
       let stride = (await getOperandsValue(context[1][7]))[0];               // 2
       let stride1 = (await getOperandsValue(context[1][8]))[0];
       let act = (await getOperandsValue(context[1][9]))[0];                  // 1
       await saveToLocalFile(weightFile);
       await saveToLocalFile(biasFile);
      let caseSample = `
         describe('CTS Real Model Test', function() {
          const assert = chai.assert;
          const nn = navigator.ml.getNeuralNetworkContext();
          it('Check result for conv_2d by convs', async function() {
            this.timeout(120000);
            let model = await nn.createModel(options);
            let operandIndex = 0;
            let op1_value;
            let op4_expect;
            await fetch('./realmodel/testcase/res/${inputName}').then((res) => {
              return res.json();
            }).then((text) => {
              let file_data = new Float32Array(text.length);
              for (let j in text) {
                let b = parseFloat(text[j]);
                file_data[j] = b;
              }
              op1_value = file_data;
            });
            await fetch('./realmodel/testcase/res/${outputName}').then((res) => {
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
            let type2 = {type: nn.TENSOR_FLOAT32, dimensions: [${bias}}]};
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
            await fetch('./realmodel/testcase/res/${weightFile}').then((res) => {
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
            await fetch('./realmodel/testcase/res/${biasFile}').then((res) => {
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
        });
         `;
         await saveCaseToLocal(caseSample, `conv${weightFile}.js`)
     } break;
     case 22: {
      let shape = (await getOperandsValue(context[1][1]))[1];
      let shape1 = (await getOperandsValue(context[1][1]))[0];
      let caseSample = `
         describe('CTS Real Model Test', function() {
          const assert = chai.assert;
          const nn = navigator.ml.getNeuralNetworkContext();
          it('Check result for reshape by ${arguments[0][1]}', async function() {
              this.timeout(120000);
              let model = await nn.createModel(options);
              let operandIndex = 0;
              let op1_value;
              let op3_expect;
              await fetch('./realmodel/testcase/res/${inputName}').then((res) => {
              return res.json();
              }).then((text) => {
              let file_data = new Float32Array(text.length);
              for (let j in text) {
                  let b = parseFloat(text[j]);
                  file_data[j] = b;
              }
              op1_value = file_data;
              });
              await fetch('./realmodel/testcase/res/${outputName}').then((res) => {
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
              let type1 = {type: nn.TENSOR_INT32, dimensions: [${shape}]};
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
      await saveCaseToLocal(caseSample, `reshape${inputName}.js`)
     }break
   }
 }

 async function generateCase(input) {
   if (input.hasOwnProperty('operations')) {
     for (let i = 0; i < input.operations.length; i++) {
       let caseType = input.operations[i][0];
       splitContext(input.operations[i]);
     }
   }
 }