
class PoseNet{
    constructor(tfmodel, backend, version, outputstride, input, type){
        this._tfmodel = tfmodel;
        this._model = null;
        this._compilation;
        this._execution;
        this._tensorIds = [];
        this._operandIndex = 0;
        this._version = version;
        this._outputstride = outputstride;
        this._inputs = input;
        this._shape = [];
        this._outputs = [];
        this._type = type;
        if (typeof backend !== 'undefined') {
            this._backend = backend;
        } 
        else {
            if (nnNative) {
                this._backend = 'WebML';
            } 
            else {
                this._backend = 'WASM';
            }
        }
        if (this._backend === 'WebML') {
            if (nnNative === null) {
                throw Error('Fails to initialize neural network context');
            }
            this._nn = nnNative;
        }
        else if (this._backend === 'WASM' || this._backend === 'WebGL2') {
            this._nn = nnPolyfill;
        }
    }
    async createCompiledModel(){
        let options = {};
        if(this._backend === 'WebGL2'){
            options.useWebGL2 = true;
        }
        this._model = await this._nn.createModel(options);
        await this._addTensorOperands();
        this._addOpsAndParams();
        
        await this._model.finish();
        this._compilation = await this._model.createCompilation();
        this._compilation.setPreference(this._nn.PREFER_FAST_SINGLE_ANSWER);
        await this._compilation.finish();
        this._execution = await this._compilation.createExecution();
    }

    async compute_single(inputTensor, heatmapTensor, offsetTensor){
        this._execution.setInput(0, inputTensor);
        this._execution.setOutput(0, heatmapTensor);
        this._execution.setOutput(1, offsetTensor);
        let error = await this._execution.startCompute();
        if(error){
            return error;
        }
        return 'success';
    }

    async compute_multi(inputTensor, heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd){
        this._execution.setInput(0, inputTensor);
        this._execution.setOutput(0, heatmapTensor);
        this._execution.setOutput(1, offsetTensor);
        this._execution.setOutput(2, displacement_fwd);
        this._execution.setOutput(3, displacement_bwd);
        let error = await this._execution.startCompute();
        if(error){
            return error;
        }
        return 'success';
    }


    async _addTensorOperands(){
        this._tfmodel = toOutputStridedLayers(this._tfmodel, this._outputstride);
        let dimension_out;
        let dimension_in = this._inputs;
        const type = this._nn.TENSOR_FLOAT32;
        for(let i in this._tfmodel){
            let dimension = [];
            let weights = [];
            let dimension_bias = [];
            let bias = [];
            if(this._tfmodel[i]["convType"] === "conv2d"){
                await getDimensionData("conv2d", this._version, i).then(function(data){
                    dimension.push(resize(data[0]));
                    weights.push(new Float32Array(transpose_weights(data[1], data[0])));
                    dimension_bias.push(data[2]);
                    bias.push(data[3]);
                });
                dimension_out = this._calculateOutput(dimension_in, dimension[0], this._tfmodel[i]["stride"], "conv2d");
                dimension_in = dimension_out;
                this._outputs.push(this._operandIndex);
                this._model.addOperand({type: type, dimensions: dimension_out});
                this._operandIndex++;
            }
            if(this._tfmodel[i]["convType"] === "separableConv"){
                let data = await getDimensionData("separableConv", this._version, i);
                if(this._tfmodel[i].rate == 1){
                    dimension.push(resize(data[0][0]));
                    dimension.push(resize(data[0][1]));
                    weights.push(new Float32Array(transpose_weights(data[1][0], data[0][0])));
                    weights.push(new Float32Array(transpose_weights(data[1][1], data[0][1])));
                    dimension_bias.push(data[2][0]);
                    dimension_bias.push(data[2][1]);
                    bias.push(data[3][0]);
                    bias.push(data[3][1]);
                }else{
                    let dilationData = dilationWeights(new Float32Array(transpose_weights(data[1][0], data[0][0])), 
                                                        resize(data[0][0]), this._tfmodel[i].rate);
                    dimension.push(dilationData[0]);
                    dimension.push(resize(data[0][1]));
                    weights.push(dilationData[1]);
                    weights.push(new Float32Array(transpose_weights(data[1][1], data[0][1])));
                    dimension_bias.push(data[2][0]);
                    dimension_bias.push(data[2][1]);
                    bias.push(data[3][0]);
                    bias.push(data[3][1]);
                }

                if(this._tfmodel[i].rate == 1){
                    dimension_out = this._calculateOutput(dimension_in, dimension[0], this._tfmodel[i]["stride"], "depthwise");
                }
                else{
                    dimension_out = this._calculateOutput(dimension_in, resize(data[0][0]), this._tfmodel[i]["stride"], "depthwise");
                }
                dimension_in = dimension_out;
                this._outputs.push(this._operandIndex);
                this._model.addOperand({type:type, dimensions: dimension_out});
                this._operandIndex++;
                dimension_out = this._calculateOutput(dimension_in, dimension[1], 1, "pointwise");
                dimension_in = dimension_out;
                this._outputs.push(this._operandIndex);
                this._model.addOperand({type:type, dimensions: dimension_out});
                this._operandIndex++;
            }
            for(let j in dimension){
                let tensorId = this._operandIndex++;
                let tensorType = {type: type, dimensions: dimension[j]};
                this._shape.push(dimension[j]);
                this._model.addOperand(tensorType);
                this._model.setOperandValue(tensorId, weights[j]);
                tensorId = this._operandIndex++;
                let tensorType_bias = {type: type, dimensions: dimension_bias[j]};
                this._model.addOperand(tensorType_bias);
                this._model.setOperandValue(tensorId, bias[j]);

            }      
        }
        //insert output layer
        let tensorType, tensorType_bias;
        let value, value_bias;
        let shape_weights, shape_bias;
        await getOutputLayer("heatmap", this._version).then(function(data){
            tensorType = {type: type, dimensions: resize(data[0])};
            tensorType_bias = {type: type, dimensions: data[2]};
            value = new Float32Array(transpose_weights(data[1], data[0]));
            value_bias = data[3];
       }); 
        let tensorId = this._operandIndex++;
        this._model.addOperand(tensorType);
        this._model.setOperandValue(tensorId, value);
        tensorId = this._operandIndex++;
        this._model.addOperand(tensorType_bias);
        this._model.setOperandValue(tensorId, value_bias);
        
        
        await getOutputLayer("offset", this._version).then(function(data){
            tensorType = {type: type, dimensions: resize(data[0])};
            tensorType_bias = {type: type, dimensions: data[2]};
            value = new Float32Array(transpose_weights(data[1], data[0]));
            value_bias = data[3];
       }); 
        tensorId = this._operandIndex++;
        this._model.addOperand(tensorType);
        this._model.setOperandValue(tensorId, value);
        tensorId = this._operandIndex++;
        this._model.addOperand(tensorType_bias);
        this._model.setOperandValue(tensorId, value_bias);
        
        if(this._type === "Multiperson"){       
            await getOutputLayer("displacement_fwd", this._version).then(function(data){
                tensorType = {type: type, dimensions: resize(data[0])};
                tensorType_bias = {type: type, dimensions: data[2]};
                value = new Float32Array(transpose_weights(data[1], data[0]));
                value_bias = data[3];
            }); 
            tensorId = this._operandIndex++;
            this._model.addOperand(tensorType);
            this._model.setOperandValue(tensorId, value);
            tensorId = this._operandIndex++;
            this._model.addOperand(tensorType_bias);
            this._model.setOperandValue(tensorId, value_bias);
           
            await getOutputLayer("displacement_bwd", this._version).then(function(data){
                tensorType = {type: type, dimensions: resize(data[0])};
                tensorType_bias = {type: type, dimensions: data[2]};
                value = new Float32Array(transpose_weights(data[1], data[0]));
                value_bias = data[3];
            }); 
            tensorId = this._operandIndex++;
            this._model.addOperand(tensorType);
            this._model.setOperandValue(tensorId, value);
            tensorId = this._operandIndex++;
            this._model.addOperand(tensorType_bias);
            this._model.setOperandValue(tensorId, value_bias);
        }
        
        //Set model input and output layer
        let heatmap_dimension = [1, (this._inputs[1]-1)/this._outputstride+1, (this._inputs[2]-1)/this._outputstride+1, 17];
        let heatmap = {type: type, dimensions: heatmap_dimension};
        let offset_dimension = [1, (this._inputs[1]-1)/this._outputstride+1, (this._inputs[2]-1)/this._outputstride+1, 34];
        let offset = {type: type, dimensions: offset_dimension};

        let input = {type:type, dimensions: this._inputs};
        let tensorId_input = this._operandIndex++;
        this._model.addOperand(input);
        let tensorId_output = [];
        tensorId_output.push(this._operandIndex++);
        this._model.addOperand(heatmap);
        tensorId_output.push(this._operandIndex++);
        this._model.addOperand(offset);
        if(this._type === "Multiperson"){
            let displacement_fwd_dimension = [1, (this._inputs[1]-1)/this._outputstride+1, (this._inputs[2]-1)/this._outputstride+1, 32];
            let displacement_bwd_dimension = [1, (this._inputs[1]-1)/this._outputstride+1, (this._inputs[2]-1)/this._outputstride+1, 32];
            let displacement_fwd = {type:type, dimensions:displacement_fwd_dimension};
            let displacement_bwd = {type:type, dimensions:displacement_bwd_dimension};
            tensorId_output.push(this._operandIndex++);
            this._model.addOperand(displacement_fwd);
            tensorId_output.push(this._operandIndex++);
            this._model.addOperand(displacement_bwd);
        }
        this._model.identifyInputsAndOutputs([tensorId_input], tensorId_output);
    }

    _calculateOutput(input_dimension, shape, stride, layer){
        var padding = 1;
        var output_dimension;
        if(layer === "conv2d"){
            output_dimension = [1, Math.floor((input_dimension[1]-shape[1]+2)/stride+1), 
                                Math.floor((input_dimension[2]-shape[2]+2)/stride+1), shape[0]];
        }
        else if(layer === "depthwise"){
            output_dimension = [1, Math.floor((input_dimension[1]-shape[1]+2)/stride+1), 
                                Math.floor((input_dimension[2]-shape[2]+2)/stride+1), input_dimension[3]];
        }
        else{
            output_dimension = [1, Math.floor((input_dimension[1]-shape[1]+0)/stride+1), 
                                Math.floor((input_dimension[2]-shape[2]+0)/stride+1), shape[0]];
        }
        return output_dimension;
    }


    _addScalarInt32(value){
        const scalarInt32Type = {type: this._nn.INT32};
        let index = this._operandIndex++;
        this._model.addOperand(scalarInt32Type);
        this._model.setOperandValue(index, new Int32Array([value]));
        return index;
    }
    _addScalarFloat32(value) {
        const scalarInt32Type = {type: this._nn.FLOAT32};
        let index = this._operandIndex++;
        this._model.addOperand(scalarInt32Type);
        this._model.setOperandValue(index, new Float32Array([value]));
        return index;
  }
    
    _addOpsAndParams(){
        let index = 0;
        for(let i in this._tfmodel){
            if(this._tfmodel[i].convType=="conv2d"){
                let inputs = [];
                inputs.push(this._model._inputs[0]);
                let outputs = this._outputs[index];
                let paddingCode = 1;
                inputs.push(outputs+1);
                inputs.push(outputs+2);
                inputs.push(this._addScalarInt32(paddingCode));
                inputs.push(this._addScalarInt32(this._tfmodel[i].stride));
                inputs.push(this._addScalarInt32(this._tfmodel[i].stride));
                let fuseCode = 3; //relu6 activition function
                inputs.push(this._addScalarInt32(fuseCode));
                let opType = this._nn.CONV_2D;
                this._model.addOperation(opType, inputs, [outputs]);
            }
            else if(this._tfmodel[i].convType == "separableConv"){
                //depthwise conv
                let inputs = [];
                inputs.push(this._outputs[index]);
                index++;
                let outputs = this._outputs[index];
                let paddingCode = 1; //SAME
                inputs.push(outputs+2);
                inputs.push(outputs+3);
                inputs.push(this._addScalarInt32(paddingCode));
                inputs.push(this._addScalarInt32(this._tfmodel[i].stride));
                inputs.push(this._addScalarInt32(this._tfmodel[i].stride));
                inputs.push(this._addScalarInt32(1));
                let fuseCode = 3;
                inputs.push(this._addScalarInt32(fuseCode));
                let opType = this._nn.DEPTHWISE_CONV_2D;
                this._model.addOperation(opType, inputs, [outputs]);

                //pointwise conv
                let inputs_point = [];
                inputs_point.push(this._outputs[index]);
                index++;
                let outputs_point = this._outputs[index];
                paddingCode = 1;
                inputs_point.push(outputs_point+3);
                inputs_point.push(outputs_point+4);
                inputs_point.push(this._addScalarInt32(paddingCode));
                inputs_point.push(this._addScalarInt32(1));
                inputs_point.push(this._addScalarInt32(1));
                fuseCode = 3;
                inputs_point.push(this._addScalarInt32(fuseCode));
                opType = this._nn.CONV_2D;
                this._model.addOperation(opType, inputs_point, [outputs_point]);
            }
        }

        if(this._type === "Multiperson"){
            let paddingCode = 1;
            let index = 5;
            for(let i = 0; i<4; i++){
                let inputs = [];
                inputs.push(this._outputs[this._outputs.length-1]);
                inputs.push(this._outputs[this._outputs.length-1]+index);
                index++;
                inputs.push(this._outputs[this._outputs.length-1]+index); 
                index++;
                inputs.push(this._addScalarInt32(paddingCode));
                inputs.push(this._addScalarInt32(1));
                inputs.push(this._addScalarInt32(1));
                let fuseCode = 0;
                inputs.push(this._addScalarInt32(fuseCode));
                let opType = this._nn.CONV_2D;
                let outputs = this._model._outputs[i];
                this._model.addOperation(opType, inputs, [outputs]);
            }

        }
        else{
            let paddingCode = 1;
            let index = 5;
            for(let i = 0; i<2; i++){
                let inputs = [];
                inputs.push(this._outputs[this._outputs.length-1]);
                inputs.push(this._outputs[this._outputs.length-1]+index);
                index++;
                inputs.push(this._outputs[this._outputs.length-1]+index); 
                index++;
                inputs.push(this._addScalarInt32(paddingCode));
                inputs.push(this._addScalarInt32(1));
                inputs.push(this._addScalarInt32(1));
                let fuseCode = 0;
                inputs.push(this._addScalarInt32(fuseCode));
                let opType = this._nn.CONV_2D;
                let outputs = this._model._outputs[i];
                this._model.addOperation(opType, inputs, [outputs]);
            }
        }
    }
}


