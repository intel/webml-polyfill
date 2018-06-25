function loadImage(imagePath, ctx) {
  const image = new Image();
  const promise = new Promise((resolve, reject) => {
    image.crossOrigin = '';
    image.onload = () => {
        ctx.drawImage(image, 0, 0);
        resolve(image);
    };
  });
  image.src = imagePath;
  return promise;
}

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
    await loadImage("download.png", ctx);
    prepareInputTensor(inputTensor, canvas, 16, [513, 513, 3]);
    var result = await net.compute_single(inputTensor, heatmapTensor, offsetTensor);
    let final_pos = decodeSinglepose(heatmapTensor, offsetTensor, [33, 33, 17], 16);
    final_pos.forEach((pose)=>{
        if(pose.score >= 0.5){
            drawKeypoints(pose.keypoints, 0.5, ctx);
            drawSkeleton(pose.keypoints, 0.5, ctx);
        }
    });   
}
//test_single();


async function test_multiple(){
    var net = new PoseNet(Architecture, 'WebGL2', 1.01, 16, [1, 513, 513, 3], "Multiperson");
    await net.createCompiledModel();

    let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    let heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
    let offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
    let displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
    let displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
    var canvas = document.getElementById("canvas_2");
    var ctx = canvas.getContext("2d");
    
    await loadImage("download.png", ctx);
    prepareInputTensor(inputTensor,canvas, 16, [513, 513, 3]);
    var start = performance.now();
    var result = await net.compute_multi(inputTensor, heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd);
    console.log("execution time: ", performance.now()-start);
    var poses = decodeMultiPose(heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd, 
                        16, 15, 0.5, 20, [33, 33, 17]); 
    poses.forEach((pose)=>{
        if(pose.score >= 0.5){
            drawKeypoints(pose.keypoints, 0.5, ctx);
            drawSkeleton(pose.keypoints, 0.5, ctx);
        }
    });
}


function getInput(inputElement){
    var reader = new FileReader();
    const promise = new Promise((resolve, reject)=>{
        reader.onload = function(e){
            resolve(e.target.result);
        }
        reader.readAsDataURL(inputElement);
    });
    return promise;
}




async function DrawSingleandMulti(){
    var net = new PoseNet(Architecture, 'WebGL2', 1.01, 16, [1, 513, 513, 3], "Multiperson");
    await net.createCompiledModel();
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");
    var inputElement = document.getElementById('image').files[0];
    var reader = new FileReader();
    var x = await getInput(inputElement);
    await loadImage(x, ctx);
    let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    let heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
    let offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
    let displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
    let displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
    prepareInputTensor(inputTensor,canvas, 16, [513, 513, 3]);
    var result = await net.compute_multi(inputTensor, heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd);
    var poses = decodeMultiPose(heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd, 
                        16, 15, 0.5, 20, [33, 33, 17]); 
    console.log(poses);
    poses.forEach((pose)=>{
        if(pose.score >= 0.5){
            drawKeypoints(pose.keypoints, 0.5, ctx);
            drawSkeleton(pose.keypoints, 0.5, ctx);
        }
    });
}


//test_multiple();