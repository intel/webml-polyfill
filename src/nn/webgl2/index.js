import webgl2 from './WebGL2'
import Tensor from './Tensor'
import Layer from './Layer'
import Model from './Model'
import layer from './layers'
import shader from './webgl'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import squeeze from 'ndarray-squeeze'

const supportWebGL2 = webgl2.supportWebGL2;
// const webmlGL = { Tensor, Layer, webgl2, layer, shader, ndarray, ops, squeeze }
// export { webmlGL as default }

export { supportWebGL2 as default };
