const HEATMAP_TENSOR_SIZE = 33*33*17;
const OFFSET_TENSOR_SIZE = 33*33*34;
const DISPLACEMENT_FWD_SIZE = 33*33*32;
const DISPLACEMENT_BWD_SIZE = 33*33*32;

async function test_multiple(){
    var net = new PoseNet(mobileNet100Architecture, 'WebGL2', 1.01, 16, [1, 513, 513, 3], "Multiperson");
    await net.createCompiledModel();

    let inputTensor = new Float32Array(INPUT_TENSOR_SIZE);
    let heatmapTensor = new Float32Array(HEATMAP_TENSOR_SIZE);
    let offsetTensor = new Float32Array(OFFSET_TENSOR_SIZE);
    let displacement_fwd = new Float32Array(DISPLACEMENT_FWD_SIZE);
    let displacement_bwd = new Float32Array(DISPLACEMENT_BWD_SIZE);
    
    let canvas_multi = document.getElementById("canvas_2");
    let ctx_multi = canvas_multi.getContext("2d");
    
    let canvas_single = document.getElementById("canvas");
    let ctx_single = canvas_single.getContext("2d");
    
    ctx_multi.clearRect(0, 0, canvas_multi.width, canvas_multi.height);
    ctx_single.clearRect(0, 0, canvas_single.width, canvas_single.height);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctx_multi);
    await loadImage("https://storage.googleapis.com/tfjs-models/assets/posenet/tennis_in_crowd.jpg", ctx_single);
    prepareInputTensor(inputTensor,canvas, 16, [513, 513, 3]);
    let start = performance.now();
    let result = await net.compute_multi(inputTensor, heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd);
    console.log("execution time: ", performance.now()-start);
    let poses = decodeMultiPose(heatmapTensor, offsetTensor, displacement_fwd, displacement_bwd, 
                        16, 15, 0.5, 20, [33, 33, 17]); 
    poses.forEach((pose)=>{
        if(pose.score >= 0.5){
            drawKeypoints(pose.keypoints, 0.5, ctx_multi);
            drawSkeleton(pose.keypoints, 0.5, ctx_multi);
        }
    });

    let poses_single = decodeSinglepose(heatmapTensor, offsetTensor, [33, 33, 17], 16);
    poses_single.forEach((pose)=>{
        if(pose.score >= 0.5){
            drawKeypoints(pose.keypoints, 0.5, ctx_single);
            drawSkeleton(pose.keypoints, 0.5, ctx_single);
        }
    });   
}

test_multiple();
function getInput(inputElement){
    let reader = new FileReader();
    const promise = new Promise((resolve, reject)=>{
        reader.onload = function(e){
            resolve(e.target.result);
        }
        reader.readAsDataURL(inputElement);
    });
    return promise;
}

async function DrawSingleandMulti(){
    let util = new Utils();
    await util.init('WebGL2');
    await util.predict();

}

// function main(){
//     let utils = new Utils();
//     const backend = document.getElementById('backend');
//     const wasm = document.getElementById('wasm');
//     const webgl = document.getElementById('webgl');
//     const webml = document.getElementById('webml');
//     let currentBackend = '';

//     function updateBackend(){
//         currentBackend = utils.model._backend;
//         if(getUrlParams('api_info') === 'true'){
//             backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
//         }
//         else{
//             backend.innerHTML = currentBackend;
//         }
//     }

//     function changeBackend(newBackend){
//         if(currentBackend === newBackend){
//             return;
//         }
//         backend.innerHTML = 'Setting...';
//         setTimeout(() => {
//             utils.init(newBackend).then(() => {
//                 updateBackend();
//                 utils.predict();
//             });
//         }, 10);
//     }

    
//     if (nnNative) {
//         webml.setAttribute('class', 'dropdown-item');
//         webml.onclick = function (e) {
//             changeBackend('WebML');
//         }
//     }

//     if (nnPolyfill.supportWebGL2) {
//         webgl.setAttribute('class', 'dropdown-item');
//         webgl.onclick = function(e) {
//             changeBackend('WebGL2');
//         }
//     }

//     if (nnPolyfill.supportWasm) {
//         wasm.setAttribute('class', 'dropdown-item');
//         wasm.onclick = function(e) {
//             changeBackend('WASM');
//         }
//     }

// }

