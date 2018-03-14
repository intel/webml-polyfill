import Tensor from './Tensor'
import Layer from './Layer'
import webgl2 from './WebGL2'
import * as layer from './layer'
import * as shader from './webgl'


import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import squeeze from 'ndarray-squeeze'

const supportWebGL2 = webgl2.supportWebGL2
// export { Tensor, Layer, webgl2, GPU_SUPPORT, layer, shader, ndarray, ops, squeeze }
// const webmlGPU = { Tensor, Layer, webgl2, GPU_SUPPORT, layer, shader, ndarray, ops, squeeze }
// export { webmlGPU as default }

export { supportWebGL2 as default };
