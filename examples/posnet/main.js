async function test(){
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
        prepareInputTensor(inputTensor, canvas, 16);
        console.log(inputTensor);
        var result = await net.compute_single(inputTensor, heatmapTensor, offsetTensor);
        console.log(sigmoid(heatmapTensor));
        //decodeSinglepose(heatmapTensor, offsetTensor, [33, 33, 17], 16);
    };
    img.src = 'download.png';
    
}

test();