| Op Type | WASM | WebGL | NNAPI | MPS | BNNS | clDNN |
|----|------|--------|-------|-----|-----|-----|
| ADD | yes | yes | yes | yes | yes | yes |
| AVERAGE_POOL_2D | yes | yes| yes | yes | yes | yes |
| CONCATENATION | yes | yes| yes | yes | yes | yes |
| CONV_2D | yes | yes| yes | yes | yes | yes |
| ATROUS_CONV_2D | yes | yes | [#415](https://github.com/intel/webml-polyfill/issues/415) | [#360](https://github.com/intel/webml-polyfill/issues/360) | [#359](https://github.com/intel/webml-polyfill/issues/359) | yes |
| DEPTHWISE_CONV_2D | yes | yes| yes | yes | [#368](https://github.com/intel/webml-polyfill/issues/368) | yes |
| ATROUS_DEPTHWISE_CONV_2D | yes | yes | [#415](https://github.com/intel/webml-polyfill/issues/415) | [#360](https://github.com/intel/webml-polyfill/issues/360) | [#359](https://github.com/intel/webml-polyfill/issues/359) | yes |
| MAX_POOL_2D |  yes | yes| yes | yes | yes | yes |
| MUL |  yes | yes | yes | yes | yes | yes |
| RESHAPE |  yes | yes| yes | yes | yes | yes |
| RESIZE_BILINEAR | yes | yes | yes | [#339](https://github.com/intel/webml-polyfill/issues/339) | [#340](https://github.com/intel/webml-polyfill/issues/340) | yes |
| SOFTMAX |  yes | yes | yes | yes | yes | yes |
| FULLY_CONNECTED | yes | yes | yes | yes | yes | yes |
