async function test_single(){
    var net = new PoseNet(Architecture, 'WebGL2', 1.01, 16, [1, 513, 513, 3], "Singleperson");
    await net.createCompiledModel();
    
    let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    let heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
    let offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
    let displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
    let displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var img = new Image();
    img.onload = async function(){
        ctx.drawImage(img, 0, 0);
        let pixels = ctx.getImageData(0, 0, 513, 513);
        prepareInputTensor(inputTensor, canvas, 16, [513, 513, 3]);
        var result = await net.compute_single(inputTensor, heatmapTensor, offsetTensor);
        let final_pos = decodeSinglepose(heatmapTensor, offsetTensor, [33, 33, 17], 16);
        console.log(final_pos);
    };
    img.src = 'download.png';
    
}

// test_single();

async function test_multiple(){
    var net = new PoseNet(Architecture, 'WebGL2', 1.01, 16, [1, 513, 513, 3], "Multiperson");
    await net.createCompiledModel();

    let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    let heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
    let offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
    let displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
    let displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var img = new Image();
    img.onload = async function(){
        ctx.drawImage(img, 0, 0);
        let pixels = ctx.getImageData(0, 0, 513, 513);
        prepareInputTensor(inputTensor, canvas, 16, [513, 513, 3]);
        var result = await net.compute_multi(inputTensor, heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd);
        decodeMultiPose(heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd, 
                        16, 15, 0.5, 20, [33, 33, 17]);
    };
    img.src = 'download.png';
}
test_multiple();